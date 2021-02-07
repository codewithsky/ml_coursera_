function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

%Suppose I have a vector and I want to know which elements of the vector 
% are greater than 3.14.
%> v = [ 7.2; 3.04; 1; 3.16; -5 ]
% result = (v > 3.14)

%-------------------------------------------------------------------------

h= sigmoid(X*theta);
p = (h >= 0.5);


% =========================================================================


end
