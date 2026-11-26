import Foundation
import MLX
import MLXNN

public class DINOv2MLP: Module {
    public let fc1: Linear
    public let fc2: Linear
    
    public init(inFeatures: Int, hiddenFeatures: Int, outFeatures: Int) {
        self.fc1 = Linear(inFeatures, hiddenFeatures)
        self.fc2 = Linear(hiddenFeatures, outFeatures)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = fc1(x)
        out = gelu(out) // DINOv2 uses GELU
        out = fc2(out)
        return out
    }
}
