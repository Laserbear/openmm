/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "CudaKernels.h"
#include "CudaForceInfo.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/common/ContextSelector.h"
#include "CommonKernelSources.h"
#include "CudaBondedUtilities.h"
#include "CudaExpressionUtilities.h"
#include "CudaIntegrationUtilities.h"
#include "CudaNonbondedUtilities.h"
#include "CudaKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <assert.h>
#include <cstdlib>
#include <string>

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

static void getCudaPmeParameters(CudaContext& cu, bool& usePmeQueue, bool& useFixedPointChargeSpreading) {
    usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    useFixedPointChargeSpreading = cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces;
}

void CudaCalcForcesAndEnergyKernel::initialize(const System& system) {
}

// ---------------- CUDA Graphs support (optional) ----------------
#if CUDA_VERSION >= 10000
namespace {
inline bool envUseGraphs() {
    const char* e = std::getenv("OPENMM_CUDA_USE_GRAPHS");
    return (e != nullptr && std::string(e) == "1");
}
}

struct CudaGraphsState {
    bool enabled = false;
    bool captured = false;
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;
    int interactingTilesSize = -1;
    int singlePairsSize = -1;
};
#endif
// ---------------------------------------------------------------

void CudaCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    cu.setForcesValid(true);
    ContextSelector selector(cu);
    cu.clearAutoclearBuffers();
    cu.updateGlobalParamValues();
    for (auto computation : cu.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    cu.setComputeForceCount(cu.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = cu.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double CudaCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    ContextSelector selector(cu);
    cu.getBondedUtilities().computeInteractions(groups);
    // Optionally submit nonbonded via a CUDA Graph (capture the hot launch sequence).
#if CUDA_VERSION >= 10000
    static thread_local CudaGraphsState graphs;
    if (!graphs.enabled)
        graphs.enabled = envUseGraphs();
    bool usedGraph = false;
    if (graphs.enabled) {
        auto& nb = cu.getNonbondedUtilities();
        int sigTiles = (int) nb.getInteractingTiles().getSize();
        int sigPairs = (int) nb.getSinglePairs().getSize();
        bool needCapture = !graphs.captured || sigTiles != graphs.interactingTilesSize || sigPairs != graphs.singlePairsSize || graphs.exec == nullptr;
        if (needCapture) {
            if (graphs.exec) {
                cuGraphExecDestroy(graphs.exec);
                graphs.exec = nullptr;
            }
            if (graphs.graph) {
                cuGraphDestroy(graphs.graph);
                graphs.graph = nullptr;
            }
            CUstream s = cu.getCurrentStream();
            CUresult r = cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_RELAXED);
            if (r == CUDA_SUCCESS) {
                nb.computeInteractions(groups, includeForces, includeEnergy);
                CUgraph g = nullptr;
                r = cuStreamEndCapture(s, &g);
                if (r == CUDA_SUCCESS && g != nullptr) {
                    graphs.graph = g;
                    CUgraphExec exec = nullptr;
                    r = cuGraphInstantiate(&exec, g, 0);
                    if (r == CUDA_SUCCESS) {
                        graphs.exec = exec;
                        graphs.captured = true;
                        graphs.interactingTilesSize = sigTiles;
                        graphs.singlePairsSize = sigPairs;
                        usedGraph = true;
                    }
                }
            }
            if (!usedGraph)
                nb.computeInteractions(groups, includeForces, includeEnergy);
        }
        else {
            if (graphs.exec) {
                cuGraphLaunch(graphs.exec, cu.getCurrentStream());
                usedGraph = true;
            }
            if (!usedGraph)
                nb.computeInteractions(groups, includeForces, includeEnergy);
        }
    }
    else
#endif
        cu.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    double sum = 0.0;
    for (auto computation : cu.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cu.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += cu.reduceEnergy();
    if (!cu.getForcesValid())
        valid = false;
    return sum;
}

void CudaCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getCudaPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, usePmeQueue, false, useFixedPointChargeSpreading, cu.getPlatformData().useCpuPme);
}

void CudaCalcConstantPotentialForceKernel::initialize(const System& system, const ConstantPotentialForce& force) {
    bool usePmeQueue, useFixedPointChargeSpreading;
    getCudaPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, false, useFixedPointChargeSpreading);
}
