import Foundation
import MLX
import MLXNN

public class TriplaneUpsampleNetwork: Module {
    public let upsample: ConvTransposed2d
    
    public init(inChannels: Int = 1024, outChannels: Int = 40) {
        // In MLX, ConvTranspose2d takes [B, H, W, C]
        self.upsample = ConvTransposed2d(
            inputChannels: inChannels, 
            outputChannels: outChannels, 
            kernelSize: [2, 2], 
            stride: [2, 2]
        )
    }
    
    public func callAsFunction(_ triplanes: MLXArray) -> MLXArray {
        // PyTorch TripoSR outputs Triplanes as [B, 3, Ci, Hp, Wp]
        // But in our MLX port, we can maintain the MLX native format: [B, 3, Hp, Wp, Ci]
        let B = triplanes.shape[0]
        let Ci = triplanes.shape[4]
        let Hp = triplanes.shape[2]
        let Wp = triplanes.shape[3]
        
        // Flatten B and 3 for the ConvTranspose: [(B * 3), Hp, Wp, Ci]
        var out = triplanes.reshaped([B * 3, Hp, Wp, Ci])
        
        // Upsample
        out = upsample(out)
        
        let Ho = out.shape[1]
        let Wo = out.shape[2]
        let Co = out.shape[3]
        
        // Reshape back: [B, 3, Ho, Wo, Co]
        out = out.reshaped([B, 3, Ho, Wo, Co])
        
        return out
    }
}
