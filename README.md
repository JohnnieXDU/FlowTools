# FlowTools
An efficient tool for optical flow / scene flow analysis.

## usage
from flowtools import FlowTools

FT = FLowTools()

## visualize your flow
flowrgb = FT.vizflow(flow, viz=True)

## compute flow error (EPE)
errmap, epe = FT.epemap(flow, flow_gt)
