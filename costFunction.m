function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
 
J = 0;
grad = zeros(size(theta));

% ===========================================================
% J_0=first component of J
% J_1= second component of J
% ============================================================

J_0=-y'*log(sigmoid(X*theta));
J_1=(1-y)'*log(1-sigmoid(X*theta));
J=(J_0 - J_1)/m;

grad=(X'*(sigmoid(X*theta)-y))/m;


% =============================================================

end
