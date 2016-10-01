function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ======================CODE======================

J_0=-y'*log(sigmoid(X*theta));
J_1=(1-y)'*log(1-sigmoid(X*theta));
%non-regularized gradient
grad_0=(X'*(sigmoid(X*theta)-y))/m ;

theta(1)=0 ; %requirement for regularization
theta2= theta'*theta ; %computing the square sum of theta
%Regularized cost function
J=(J_0 - J_1)/m +lambda*(theta2)/(2*m);
%Regularized gradient
grad= grad_0 + theta*lambda/m ;

% =============================================================

end
