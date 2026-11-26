import Foundation
import MLX
import MLXNN

public class NeRFMLP: Module {
    public let layers: [Module]
    
    public init(inChannels: Int = 120, nNeurons: Int = 64, nHiddenLayers: Int = 9) {
        var modules: [Module] = []
        
        // Input layer
        modules.append(Linear(inChannels, nNeurons, bias: true))
        modules.append(SiLU())
        
        // Hidden layers
        for _ in 0..<nHiddenLayers - 1 {
            modules.append(Linear(nNeurons, nNeurons, bias: true))
            modules.append(SiLU())
        }
        
        // Output layer (density + RGB = 4)
        modules.append(Linear(nNeurons, 4, bias: true))
        
        self.layers = modules
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [..., inChannels]
        var out = x
        for layer in layers {
            if let linear = layer as? Linear {
                out = linear(out)
            } else if let silu = layer as? SiLU {
                out = silu(out)
            }
        }
        return out
    }
}
