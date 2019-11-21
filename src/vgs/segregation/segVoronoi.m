clear all; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha  = 1.0; % control gain.
dAA    = 2.0;   % distance among same types.
dAB    = 5.0;   % distance among distinct types.

% Simulation.
ROBOTS = 50;    % number of robots.
GROUPS = 5;      % number of groups.
WORLD  = 10;     % world size.
dt     = 0.01;  % time step.

if mod(ROBOTS,GROUPS) ~= 0
    fprintf('ROBOTS mod GROUPS must be 0.');
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AA(i,j) == 1 if i and j belong to the same team,  0 otherwise.
% AB(i,j) == 1 if i and j belong to distinct teams, 0 otherwise.
[i j] = meshgrid((1:ROBOTS));
gpr = GROUPS / ROBOTS;
AA  = (floor(gpr*(i-1)) == floor(gpr*(j-1)));
AB  = (floor(gpr*(i-1)) ~= floor(gpr*(j-1)));
clearvars i j;

% vectorization of dAA and dAB.
const = dAA .* AA + dAB .* AB;

q = WORLD * rand(ROBOTS, 2); % position.
v = zeros(ROBOTS, 2);        % velocity.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize all plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simulation = figure('Backingstore', 'off');
hold on;
xlim([0 WORLD]);
ylim([0 WORLD]);

% Set handlers.
color = hsv(GROUPS);
for i = (1:GROUPS)
    handler(i) = plot(nan, '.');
    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
    stop  = floor(i * ROBOTS / GROUPS);
    set(handler(i),'Color',color(i,:), 'MarkerSize', 20);
    set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
end
handler(GROUPS+1) = plot(nan, '.');
title('Press any key to start');
waitforbuttonpress;
title('Press any key to stop');
set(simulation, 'currentch', char(0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
while strcmp(get(simulation, 'currentch'), '')    
    % Relative position among all pairs [q(j:2) - q(i:2)].
    xij  = bsxfun(@minus, q(:,1)', q(:,1));
    yij  = bsxfun(@minus, q(:,2)', q(:,2));
    
    % Relative velocity among all pairs [v(j:2) - v(i:2)]..
    vxij = bsxfun(@minus, v(:,1)', v(:,1));
    vyij = bsxfun(@minus, v(:,2)', v(:,2));
    
    % Relative distance among all pairs.
    dsqr = xij.^2 + yij.^2;
    dist = sqrt(dsqr);
       
    % Control equation.
    dV = alpha .* (dist - const + 1.0 ./ dist - const ./ dsqr);
    ax = - dV .* xij ./ dist - vxij;
    ay = - dV .* yij ./ dist - vyij;
    
    % This gets rid of NaN because of division by zero.
    ax(1:ROBOTS+1:end) = 0;
    ay(1:ROBOTS+1:end) = 0;
      
    % a(i, :) -> acceleration input for robot i.
    a = [sum(ax)' sum(ay)'];
    
    % simple taylor expansion.
    q = q + v * dt + a * (0.5 * dt^2);
    v = v + a * dt;
    
    % Update data for drawing.
    for i = (1:GROUPS)
        start = floor((i - 1) * ROBOTS / GROUPS) + 1;
        stop  = floor(i * ROBOTS / GROUPS);
        set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
    end
    drawnow;
    
end

mu = zeros(GROUPS, 2);
for i = (1:GROUPS)
    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
    stop  = floor(i * ROBOTS / GROUPS);
    mu(i, :) = mean(q(start:stop,:));
end
[vx, vy] = voronoi(mu(:,1), mu(:,2));
plot(vx, vy);
title('');