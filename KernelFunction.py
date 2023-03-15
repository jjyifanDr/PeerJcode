# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
import torch
from torch.autograd import Function, Variable


  
def Matern(input1, input2, log_amp, sigma_sqrt):

	n1=n2 = input2.size(0)

	if sigma_sqrt.dim() == 1:
		transformed_input1 = input1 * sigma_sqrt
		transformed_input2 = input2 * sigma_sqrt
	elif sigma_sqrt.dim() == 2:
		transformed_input1 = torch.mm(input1.to(torch.double), sigma_sqrt.to(torch.double))
		transformed_input2 = torch.mm(input2.to(torch.double), sigma_sqrt.to(torch.double))

	output = log_amp.exp() * (transformed_input1.unsqueeze(1).repeat(1, n2, 1) * transformed_input2.unsqueeze(0).repeat(n1, 1, 1)).sum(2)
	return output


class InnerProduct(Function):

	@staticmethod
	def forward(ctx, input1, input2, log_amp, sigma_sqrt):
		#ndim = input1.size(1)
		n1=n2 = input2.size(0)

		if sigma_sqrt.dim() == 1:
			transformed_input1 = input1 * sigma_sqrt
			transformed_input2 = input2 * sigma_sqrt
		elif sigma_sqrt.dim() == 2:
			transformed_input1 = torch.mm(input1.to(torch.double), sigma_sqrt.to(torch.double))
			transformed_input2 = torch.mm(input2.to(torch.double), sigma_sqrt.to(torch.double))

		output = log_amp.exp() * (transformed_input1.unsqueeze(1).repeat(1, n2, 1) * transformed_input2.unsqueeze(0).repeat(n1, 1, 1)).sum(2)
		ctx.save_for_backward(input1, input2, log_amp, sigma_sqrt, output)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		input1, input2, log_amp, sigma_sqrt, output = ctx.saved_variables
		grad_input1 = grad_input2 = grad_log_amp = grad_sigma_sqrt = None

		ndim = input1.size(1)
		n1=n2 = input2.size(0)

		if sigma_sqrt.dim() == 1:
			sigma = sigma_sqrt ** 2
			mat_input1 = input1 * sigma
			mat_input2 = input2 * sigma
		elif sigma_sqrt.dim() == 2:
			sigma = sigma_sqrt.mm(sigma_sqrt.t())
			mat_input1 = torch.mm(input1.to(torch.double), sigma.to(torch.double))
			mat_input2 = torch.mm(input2.to(torch.double), sigma.to(torch.double))

		if ctx.needs_input_grad[0]:
			grad_input1 = grad_output.mm(mat_input2) * log_amp.exp()
		if ctx.needs_input_grad[1]:
			grad_input2 = grad_output.t().mm(mat_input1) * log_amp.exp()
		if ctx.needs_input_grad[2]:
			grad_log_amp = (grad_output * output).sum()
		if ctx.needs_input_grad[3]:
			if sigma_sqrt.dim() == 1:
				all_prod = grad_output.unsqueeze(2).repeat(1, 1, ndim) * input1.unsqueeze(1).repeat(1, n2, 1) * input2.unsqueeze(0).repeat(n1, 1, 1) * sigma_sqrt.view(1, 1, -1).repeat(n1, n2, 1)
				grad_sigma_sqrt = log_amp.exp() * 2.0 * all_prod.sum(0).sum(0)
			elif sigma_sqrt.dim() == 2:
				y_xT = torch.bmm(input2.unsqueeze(2), (input1.unsqueeze(1).repeat(1, n2, 1) * grad_output.unsqueeze(2).repeat(1, 1, ndim)).sum(0, keepdim=True).transpose(0, 1)).sum(0)
				input_outer = y_xT + y_xT.t()
				grad_sigma_sqrt = log_amp.exp() * input_outer.mm(sigma_sqrt.to(torch.double))

		return grad_input1, grad_input2, grad_log_amp, grad_sigma_sqrt


def CalKernel_(input_data):
    	
	ndim = input_data.shape[1]
	n2 = n1 = input_data.shape[0]
	input_grad = False
	param_grad = not input_grad
      
	input1 = Variable(torch.from_numpy(input_data), requires_grad=input_grad)    
	input2 = Variable(torch.from_numpy(input_data), requires_grad=input_grad)
       
	log_amp = Variable(torch.randn(1), requires_grad=False)
	sigma_sqrt = Variable(torch.tril(torch.randn(ndim, ndim)), requires_grad=param_grad)


	if sigma_sqrt.grad is not None:
		sigma_sqrt.grad.data.zero_()
        
	output1 = InnerProduct.apply(input1, input2, log_amp, sigma_sqrt)
	output1.backward(torch.ones(n1, n2))
	sigma_sqrt.grad.data.zero_()
       
	output2 = Matern(input1, input2, log_amp, sigma_sqrt)
	output2.backward(torch.ones(n1, n2))

	_tensor = torch.tensor(sigma_sqrt.grad)
    
	return _tensor   #return a tensor
