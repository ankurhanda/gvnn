local ReverseXYOrder, parent = torch.class('nn.ReverseXYOrder', 'nn.Module')

function ReverseXYOrder:__init(inputSize)
   parent.__init(self)
end

function ReverseXYOrder:updateOutput(input)
   self.output:resizeAs(input)

   self.output:select(4,1):copy(input:select(4,2))
   self.output:select(4,2):copy(input:select(4,1))

   return self.output 
end

function ReverseXYOrder:updateGradInput(input, gradOutput) 
   
   self.gradInput:resizeAs(input):zero()
   self.gradInput:select(4,1):copy(gradOutput:select(4,2))
   self.gradInput:select(4,2):copy(gradOutput:select(4,1))
 
   return self.gradInput
end
