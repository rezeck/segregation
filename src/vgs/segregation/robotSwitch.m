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
U1 = collectivePotential(q, const, alpha);

points = ginput(2);
dist = sum(bsxfun(@minus, points(1,:), q).^2, 2);
s1 = find(dist == min(dist));
dist = sum(bsxfun(@minus, points(2,:), q).^2, 2);
s2 = find(dist == min(dist));

tmp = q(s1, :); q(s1,:) = q(s2,:); q(s2,:) = tmp;
U2 = collectivePotential(q, const, alpha); U1 - U2
delete(hullHandler);
end