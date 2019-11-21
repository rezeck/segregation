function segregation_video(ROBOTS, GROUPS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha  = 5.0; % control gain.
dAA    = 3.0; % distance among same types.
dAB    = 6.0; % distance among distinct types.

% Simulation.
%ROBOTS = 150;   % number of robots.
%GROUPS = 5;    % number of groups.
WORLD  = 10;   % world size.
dt     = 0.01; % time step.

if mod(ROBOTS,GROUPS) ~= 0
    fprintf('ROBOTS mod GROUPS must be 0.');
    return
end

if (length(dAA) > 1) && (length(dAA) ~= GROUPS)
    fprintf('length(dAA) must be equal to GROUPS');
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

% vectorization of dAA and dAB.
if (length(dAA) == 1)
    const = dAA .* AA + dAB .* AB;
else
    const = kron(diag(dAA), ones(ROBOTS/GROUPS)) + dAB .* AB;
end
clear i j;

q = initial_positions(ROBOTS, 2, WORLD, 0.2);
%q = WORLD * rand(ROBOTS, 2); % position.
v = zeros(ROBOTS, 2);         % velocity.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize all plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simulation = figure('Backingstore', 'off', ...
                    'units','pixels','Position', [0 0 800 600]);

hold on;
center = mean(q);
xlim([center(1) - 1.1*WORLD/2, center(1) + 1.1*WORLD/2]);
ylim([center(2) - 1.1*WORLD/2, center(2) + 1.1*WORLD/2]);

% Set handlers.
color = distinguishable_colors(GROUPS);
for i = (1:GROUPS)
    handler(i) = plot(nan, '.');
    start = floor((i - 1) * ROBOTS / GROUPS) + 1;
    stop  = floor(i * ROBOTS / GROUPS);
    set(handler(i),'Color',color(i,:), 'MarkerSize', 30);
    set(handler(i),'XData',q(start:stop,1),'YData',q(start:stop,2));
end
axis manual; axis off;

title = ['segregation2D-r', num2str(ROBOTS), 'g', num2str(GROUPS), '.avi'];
video = VideoWriter(title);
video.Quality = 100;
video.FrameRate = 30;
open(video);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while true
        
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
      
    % a(i, :) -> acceleration input for robot i.
    a = [nansum(ax)' nansum(ay)'];
    
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
    
    writeVideo(video, im2frame(print('-RGBImage')));
    
    if norm(v) < 0.04
        break;
    end
end
close(video);
close(simulation);