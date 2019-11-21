%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convex hull selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
segregation;

format long; figure(simulation);
while true
hullHandler = zeros(1, GROUPS);
chulls = cell(1, GROUPS);
for i = (1:GROUPS)
    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
    stop  = floor(i * ROBOTS / GROUPS);
    chulls{i} = convhull(q(start:stop,1), q(start:stop,2)) + start - 1;
    hullHandler(i) = plot(q(chulls{i},1), q(chulls{i},2), 'Color', color(i,:));
    set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
end

points = ginput(3);
distk = sum(bsxfun(@minus, points(1,:), q).^2, 2);
i = find(distk == min(distk));
distk = sum(bsxfun(@minus, points(2,:), q).^2, 2);
s = find(distk == min(distk));
distk = sum(bsxfun(@minus, points(3,:), q).^2, 2);
t = find(distk == min(distk));

R = alpha * ((const(i,s)-const(i,t))/dist(i,t) + (const(i,t)-const(i,s))/dist(i,s))*(dist(i,s)*dist(i,t)+1)

U1 = singlePotential(q, const, alpha, i);
tmp = q(s, :); q(s,:) = q(t,:); q(t,:) = tmp;
U2 = singlePotential(q, const, alpha, i); U2 - U1
delete(hullHandler);
end