function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

%initializing values
g = zeros(size(z));

% ===========================================================
% Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[m,n]=size(z)

for i =1:m
	for j=1:n
	
g(i,j)=1/(1+exp(-1*z(i,j)));

endfor
endfor


% =============================================================

end
