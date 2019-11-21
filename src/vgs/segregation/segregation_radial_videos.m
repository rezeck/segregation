function segregation_radial_videos(ROBOTS, GROUPS)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fact = 0.5;
alpha  = 5.0;                  % control gain.
dAB    = fact * 7.5;           % distance among distinct types.
dAA    = fact * 5.0 * (1:GROUPS);

WORLD  = 10;            % world size.
dt     = 0.01;         % time step.

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

q = initial_positions(ROBOTS, 2, WORLD, 0.2);
v = zeros(ROBOTS, 2);        % velocity.

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

title = ['radial2D-r', num2str(ROBOTS), 'g', num2str(GROUPS), '.avi'];
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
    dV = alpha .* (1.0 ./ dist - const ./ dsqr + (dist - const));
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
    
    breakFree = norm(v) < 0.1 && count_metric(q, const) < 0.001;
    if breakFree
        break;
    end
end

% compute fitted circles
rect = zeros(1, GROUPS);
for i = 1:GROUPS
   hull = convhull(q((i-1)*ROBOTS/GROUPS+1:i*ROBOTS/GROUPS, :)) ...
        + (i-1)*ROBOTS/GROUPS;
   [xc, yc, r, a] = circfit(q(hull(1:end-1), 1), q(hull(1:end-1), 2));
   rect(i) = rectangle('Position', [xc-r, yc-r, 2.0*r, 2.0*r], ...
                       'Curvature', [1, 1], 'EdgeColor', [1, 1, 1], ...
                       'LineStyle', '-.', 'LineWidth', .5);
end

% put robots on top of plot
for i = 1:GROUPS
    uistack(handler(i) , 'top');
end

% fade in circles for one second
for i = 1:video.FrameRate
    k = i / video.FrameRate;
    for j = 1:GROUPS
        set(rect(j), 'EdgeColor', color(j, :) * k + (1 - k) * [1, 1, 1]);
    end
    writeVideo(video, im2frame(print('-RGBImage')));
end

% records for five seconds
for i = 1:(5*video.FrameRate)
    writeVideo(video, im2frame(print('-RGBImage')));
end

close(video);
close(simulation);