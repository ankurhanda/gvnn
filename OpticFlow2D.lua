local OpticalFlow2D, parent = torch.class('nn.OpticalFlow2DBHWD', 'nn.Module')

--[[
   OpticalFlow2D(height, width) :
   OpticalFlow2D:updateOutput(transformMatrix)
   OpticalFlow2D:updateGradInput(transformMatrix, gradGrids)

   OpticalFlow2D will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function OpticalFlow2D:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(height, width, 3)
   for i=1,self.height do
      self.baseGrid:select(3,2):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(3,1):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
end

function OpticalFlow2D:updateOutput(optic_flow)

   local current_optic_flow  = optic_flow

   assert(PerPixelAffineMatrixParams:nDimension()==4
	  and PerPixelAffineMatrixParams:size(2)==self.height
          and PerPixelAffineMatrixParams:size(3)==self.width
          and PerPixelAffineMatrixParams:size(4)==2
          , 'please input affine per-pixel transformations (bxhxwx2)')

   local batchsize = PerPixelAffineMatrixParams:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then

      self.batchGrid:resize(batchsize, self.height, self.width, 2)

      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end

   end

   self.output:resize(batchsize, self.height, self.width, 2)
   self.output = torch.add(self.baseGrid, current_optic_flow)

   return self.output

end

function OpticalFlow2D:updateGradInput(optic_flow, _gradGrid)

   self.gradInput:resizeAs(optic_flow):zero()
   self.gradInput:copy(_gradGrid)

   return self.gradInput

end
