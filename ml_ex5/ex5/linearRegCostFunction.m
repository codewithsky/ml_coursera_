function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
errors = (h-y);
J_unreg = 1/(2*m)*(errors'*errors);

grad_unreg = 1/(m)*(X'*errors);

theta(1) = 0;
J_reg = (lambda/(2*m))*sum(theta.^2);
grad_reg = (lambda/m)*theta;

J = J_unreg + J_reg;
grad = grad_unreg + grad_reg;


% =========================================================================

grad = grad(:);

end
