function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    computeCost(X,y,theta);
    
    %non-vectorized implementation
    new_theta = zeros(1,length(theta));
    
    for j = 1:length(theta) %loop through theta_j's
        update_term = 0;
        for i = 1:m %loop through training examples
            h_theta_x_i = theta'*X(i,:)';
            err = h_theta_x_i - y(i);
            update_term = update_term + err*X(i,j);
        end %for compute update term for current theta_j
        
        new_theta(j) = theta(j) - alpha*(1/m)*update_term;
    end %for theta update
    
    theta = new_theta';
    
    %vectorized implementation
    %theta = theta - alpha/m*((X*theta-y)'*X)'; 


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end %for iterations

end %function
