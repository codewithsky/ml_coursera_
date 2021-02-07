function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%h = sigmoid stuff
%J = unregularized part
%theta(1) = 0
%J = J + regularized part
%----------------------------------------------------------------------

h = sigmoid(X*theta);
J_unregularized = -(1/m)*(y'*log(h) + (1-y)'*log(1-h));
grad_unregularized = (1/m)*X'*(h-y);

theta(1) = 0;
J_regularized = (lambda/(2*m))*theta'*theta;
grad_regularized = (lambda/m)*theta;

J = J_unregularized + J_regularized;
grad = grad_unregularized + grad_regularized;


% =============================================================

end
