function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

return;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

C_min = 0.01;
sigma_min = 0.01;
error_min = size(yval, 1);
for i = 1:size(C_vec, 2)
  for j = 1:size(sigma_vec, 2)
    c = C_vec(i);
    s = sigma_vec(j);
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
    predictions = svmPredict(model, Xval);
    errors = mean(double(predictions ~= yval));
    if (errors < error_min)
      error_min = errors;
      sigma_min = s;
      C_min = c;
    endif
  end
end

C = C_min;
sigma = sigma_min;






% =========================================================================

end
