function segregation3d_video(ROBOTS, GROUPS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha  = 5.0;  % control gain.
dAA    = 3;   % distance among same types.
dAB    = 6;   % distance among distinct types.

% Simulation.
%ROBOTS = 150;    % number of robots.
%GROUPS = 10;     % number of groups.
WORLD  = 10;     % world size.
dt     = 0.01;   % time step.

camspeed = 20;    % camera rotation speed.

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

% number of robots in one group.
nAA = ROBOTS / GROUPS;

% number of robots in distinct groups.
nAB = (GROUPS - 1) * ROBOTS / GROUPS;

q = initial_positions(ROBOTS, 3, WORLD, 0.2);
v = zeros(ROBOTS, 3);        % velocity.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize all plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simulation = figure('renderer', 'opengl', 'doublebuffer', 'on', ...
                    'units','pixels','Position', [0 0 800 600]);
hold on;

color   = zeros(ROBOTS,3);
pallete = distinguishable_colors(GROUPS);
for i = (1:GROUPS)
        start = floor((i - 1) * ROBOTS / GROUPS) + 1;
        stop  = floor(i * ROBOTS / GROUPS);
        color(start:stop, :) = repmat(pallete(i,:), nAA, 1);
end

handler = bubbleplot3(q(:, 1), q(:,2), q(:,3), ...
                      0.3*ones(ROBOTS*3, 1), color);            
                  
lighting phong;
grid on;
center = mean(q);
xlim([center(1) - 1.15*WORLD/2, center(1) + 1.15*WORLD/2]);
ylim([center(2) - 1.15*WORLD/2, center(2) + 1.15*WORLD/2]);
zlim([center(3) - 1.15*WORLD/2, center(3) + 1.15*WORLD/2]);
lim = get(gca, 'XLim'); set(gca, 'XTick', linspace(lim(1), lim(2), 4)); 
lim = get(gca, 'YLim'); set(gca, 'YTick', linspace(lim(1), lim(2), 4)); 
lim = get(gca, 'ZLim'); set(gca, 'ZTick', linspace(lim(1), lim(2), 4)); 
set(gca,'xticklabel',[], 'yticklabel', [], 'zticklabel', []);
set(gca, 'linewidth', 3.0);
view([45 45 45]);
%set(gca,'dataaspectratio',[1 1 1]);
axis manual;
fill_lgt = camlight('headlight');
set(fill_lgt, 'color', [0.3, 0.3, 0.3]);
key_lgt = camlight('right');
set(key_lgt, 'color', [0.7, 0.7, 0.7]);
camproj('perspective');
grid off;

title = ['segregation3D-r', num2str(ROBOTS), 'g', num2str(GROUPS), '.avi'];
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
    zij  = bsxfun(@minus, q(:,3)', q(:,3));
    
    % Relative velocity among all pairs [v(j:2) - v(i:2)]..
    vxij = bsxfun(@minus, v(:,1)', v(:,1));
    vyij = bsxfun(@minus, v(:,2)', v(:,2));
    vzij = bsxfun(@minus, v(:,3)', v(:,3));
    
    % Relative distance among all pairs.
    dsqr = xij.^2 + yij.^2 + zij.^2;
    dist = sqrt(dsqr);
       
    % Control equation.
    dV = alpha .* (1.0 ./ dist - const ./ dsqr + (dist - const));
    ax = - dV .* xij ./ dist - vxij;
    ay = - dV .* yij ./ dist - vyij;
    az = - dV .* zij ./ dist - vzij;
    
    % This gets rid of NaN because of division by zero.
    ax(1:ROBOTS+1:end) = 0;
    ay(1:ROBOTS+1:end) = 0;
    az(1:ROBOTS+1:end) = 0;
      
    % a(i, :) -> acceleration input for robot i.
    a = [sum(ax)' sum(ay)' sum(az)'];
    
    % simple taylor expansion.
    dq = v * dt + a * (0.5 * dt^2);
    q = q + dq;
    v = v + a * dt;
     
    % Update data for drawing.
    for i = (1:ROBOTS)
        xdata = get(handler(i), 'xdata');
        ydata = get(handler(i), 'ydata');
        zdata = get(handler(i), 'zdata');
        set(handler(i), 'xdata', xdata + dq(i, 1), ...
                        'ydata', ydata + dq(i, 2), ...
                        'zdata', zdata + dq(i, 3));
    end
    camorbit(camspeed*dt,0);
    camlight(fill_lgt, 'headlight');
    drawnow;

    writeVideo(video, im2frame(print('-RGBImage')));
    
    if norm(v) < 0.5
        break;
    end

end
close(video);
close(simulation);