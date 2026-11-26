import Foundation
import MLX
import MLXNN

public class LRMFeedForward: Module {
    public let proj1: Linear
    public let proj2: Linear
    
    public init(dim: Int, dimOut: Int? = nil, mult: Int = 4) {
        let innerDim = dim * mult
        let finalDimOut = dimOut ?? dim
        
        // In GEGLU, we project to innerDim * 2, then split and multiply
        self.proj1 = Linear(dim, innerDim * 2)
        self.proj2 = Linear(innerDim, finalDimOut)
    }
    
    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // GEGLU
        let projected = proj1(hiddenStates)
        let chunks = projected.split(parts: 2, axis: -1)
        let x = chunks[0]
        let gate = chunks[1]
        
        var out = x * gelu(gate)
        out = proj2(out)
        
        return out
    }
}
