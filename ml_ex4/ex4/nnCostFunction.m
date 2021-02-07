function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%m = the number of training examples
%n = the number of training features, including the initial bias unit.
%h = the number of units in the hidden layer - NOT including the bias unit
%r = the number of output classifications

%Part 1: Calculating J w/o Regularization
%y_matrix = eye(num_labels)(y,:);
y_matrix = (1:num_labels)==y;
a1 = [ones(m, 1), X];
z2 = a1*Theta1';

a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2]; 
%Adding 1 as first column in 
%z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26

z3 = a2*Theta2';
a3 = sigmoid(z3);
%m =5000
%J_ur = (1/m) * sum(sum((-y_matrix.*log(a3))-((1-y_matrix).*log(1-a3))));

J = -(1/m)*trace((y_matrix'*log(a3) + (1-y_matrix)'*log(1-a3)));
%disp(J);
%disp(size(Theta2));

%Part 2: Implementing Backpropogation for Theta_gra w/o Regularization

d3 = a3-y_matrix;
%disp(d3);

sg = sigmoid(z2).*(1-sigmoid(z2)); 

d2 = (d3*Theta2).* [ones(size(z2,1),1) sg];
d2 = d2(:,2:end);
%disp(d2);

Delta1 = d2'*a1;
Delta2 = d3'*a2;

%disp(size(Delta1));
%disp(size(Delta1));

Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;
%disp(size(Theta1_grad));
%disp(size(Theta2_grad));

%Part 3: Adding Regularisation term in J and Theta_grad

%Theta1 = Theta1(:,2:end); %excluding first column of bias units
%Theta2 = Theta2(:,2:end); %excluding first column of bias units
J_regularized = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
%J_regularized = (lambda/(2*m))*(trace(Theta1(:,2:end)'*Theta1(:,2:end)) + trace(Theta2(:,2:end)'*Theta2(:,2:end)));
%disp(J_regularized);

J = J + J_regularized;

Theta1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)]; %adding the bias unit 0

%disp(size(Theta1_grad)); %25   401
%disp(size(Theta2_grad)); %10   26
%disp(size(Theta1)); %25   401
%disp(size(Theta2)); %10   26

Theta1 = (lambda/m)*Theta1;
Theta2 = (lambda/m)*Theta2;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
