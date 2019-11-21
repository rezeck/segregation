function segregation_experiments(ROBOTS, GROUPS, output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha  = 2.0; % control gain.
dAB    = 7.5;  % distance among distinct types.
dAA    = 5.0*(1:GROUPS);

% Simulation.
WORLD  = 20;          % world size.
dt     = 0.005;        % time step.

if mod(ROBOTS,GROUPS) ~= 0
    fprintf('ROBOTS mod GROUPS must be 0.');
return
end

iterations  = 15000;
experiments = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AA(i,j) == 1 if i and j belong to the same team,  0 otherwise.
  % AB(i,j) == 1 if i and j belong to distinct teams, 0 otherwise.
  [i j] = meshgrid((1:ROBOTS));
rpg = ROBOTS / GROUPS;
AA = zeros(ROBOTS);
for k = 1:GROUPS
	  aux = (k-1)*rpg+1:k*rpg;
AA(aux, aux) = dAA(k);
end
gpr = GROUPS / ROBOTS;
AB  = (floor(gpr*(i-1)) ~= floor(gpr*(j-1)));
clearvars i j aux rpg gpr;

% vectorization of dAA and dAB.
const = AA + dAB .* AB;

% number of robots in one group.
nAA = ROBOTS / GROUPS;

% number of robots in distinct groups.
nAB = (GROUPS - 1) * ROBOTS / GROUPS;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize all plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulation = figure('Backingstore', 'off');
%hold on;
%xlim([0 WORLD]);
%ylim([0 WORLD]);

% Set handlers.
%color = hsv(GROUPS);

%title('Press any key to start');
%waitforbuttonpress;
%title('Press any key to stop');
%set(simulation, 'currentch', char(0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data   = zeros(experiments, iterations);
%meanAA = zeros(GROUPS, experiments, iterations);
%meanAB = zeros(GROUPS, experiments, iterations);

tic;
for exp = (1:experiments)
  %fprintf('Running experiment %d...', exp);
%expbegin = tic;
    
if mod(exp, 10) == 0
  save(output, 'data');
%save('meanAA.mat', 'meanAA');
%save('meanAB.mat', 'meanAB');
end
        
% initial states for each robot.
    
while (true)
  qi = WORLD * rand(ROBOTS, 2); % position.
  xij  = bsxfun(@minus, qi(:,1)', qi(:,1));
        yij  = bsxfun(@minus, qi(:,2)', qi(:,2));
dsqr = xij.^2 + yij.^2;
dsqr(1:ROBOTS+1:end) = NaN;
dsqr(1:ROBOTS+1:end) = NaN;
if (min(dsqr) >= 0.12^2)
  break;
end
end
    
q = qi;
v = zeros(ROBOTS, 2);        % velocity.
    
%for i = (1:GROUPS)
  %    handler(i) = plot(nan, '.');
%    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
%    stop  = floor(i * ROBOTS / GROUPS);
%    set(handler(i),'Color',color(i,:));
%    set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
%end
    
for it = (1:iterations)
    
  % Relative position among all pairs [q(j:2) - q(i:2)].
  xij  = bsxfun(@minus, q(:,1)', q(:,1));
    yij  = bsxfun(@minus, q(:,2)', q(:,2));
    
% Relative velocity among all pairs [v(j:2) - v(i:2)]..
vxij = bsxfun(@minus, v(:,1)', v(:,1));
    vyij = bsxfun(@minus, v(:,2)', v(:,2));
    
% Relative distance among all pairs.
dsqr = xij.^2 + yij.^2;
dist = sqrt(dsqr);
    
% mean(i, 1) -> mean position wrt robot i of its own group.
% mean(i, 2) -> mean position wrt robot i of its own group. 
%lmean = [sum(dist.*AA)' sum(dist.*AB)'];
%for i = (1:GROUPS)
  %    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
%    stop  = floor(i * ROBOTS / GROUPS);
%    assert(stop - start + 1 == nAA);
%    meanAA(i, exp, it) = sum(lmean(start:stop, 1)) / (nAA * (nAA - 1));
%    meanAB(i, exp, it) = sum(lmean(start:stop, 2)) / (nAA * nAB);
%end
    
% Control equation.
dV = alpha .* (1.0 ./ dist - const ./ dsqr + (dist - const));
ax = - dV .* xij ./ dist - vxij;
ay = - dV .* yij ./ dist - vyij;
    
% This gets rid of NaN because of division by zero.
ax(1:ROBOTS+1:end) = 0;
ay(1:ROBOTS+1:end) = 0;
      
% a(i, :) -> acceleration input for robot i.
a = [sum(ax)' sum(ay)'];
    
data(exp, it) = count_metric(q, const);
    
% simple taylor expansion.
q = q + v * dt + a * (0.5 * dt^2);
v = v + a * dt;
    
% Update data for drawing.
%for i = (1:GROUPS)
  %     start = floor((i - 1) * ROBOTS / GROUPS) + 1;
%     stop  = floor(i * ROBOTS / GROUPS);
%     set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
%end
%drawnow;
    
end
%fprintf('%fs\n', toc(expbegin));
    
if data(exp, end) > 0
save(['q' int2str(exp) output], 'qi');
end
end
%fprintf('Total time: %f\n', toc);
save(output, 'data');
%save('meanAA.mat', 'meanAA');
%save('meanAB.mat', 'meanAB');

%plt = figure();
%subplot(2, 1, 1);
%if experiments > 1
%    plot((1:iterations), mean(data) + 2*std(data), '--');
%    plot((1:iterations), mean(data) - 2*std(data), '--');
%    plot((1:iterations), mean(data));
%else
   %    plot((1:iterations), data);
%end
%set(gca, 'xscale','log');

%subplot(2, 1, 2);
%hold on;
%grid on;
%rAA = zeros(GROUPS, iterations);
%rAB = zeros(GROUPS, iterations);
%for i = (1:GROUPS)
  %    for j = (1:experiments)
  %        rAA(i, :) = rAA(i, :) + squeeze(meanAA(i, j, :))';
%        rAB(i, :) = rAB(i, :) + squeeze(meanAB(i, j, :))';
%    end
%end
%rAA = rAA / experiments;
%rAB = rAB / experiments;

%for i = (1:GROUPS)
  %    plot((1:iterations), rAA(i,:), 'b', (1:iterations), rAB(i,:), 'r');
%end
