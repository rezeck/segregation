function [ U ] = singlePotential( q, const, alpha, i)
%collectivePotential computes Equation 9 of the ICRA2014 paper.

% Relative position among all pairs [q(j:2) - q(i:2)].
xij  = bsxfun(@minus, q(:,1)', q(:,1));
yij  = bsxfun(@minus, q(:,2)', q(:,2));
    
% Relative distance among all pairs.
dist = sqrt(xij.^2 + yij.^2);

U = alpha * (0.5 .* (dist - const).^2 + log(dist) + const ./ dist);

% This gets rid of NaN because of division by zero.
U(1:size(U,1)+1:end) = 0;

U = sum(U(:, i));
end

