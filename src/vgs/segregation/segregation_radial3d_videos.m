function segregation_radial3d_videos(ROBOTS, GROUPS)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants (you may freely set these values).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fact = 0.6;
alpha  = 5.0;                  % control gain.
dAB    = fact * 7.5;           % distance among distinct types.
dAA    = fact * 5.0 * (1:GROUPS);

% Edit the indexes to generate specific spheres at the end of simulation.
% See SPHERE_INDEXES next to the end of the code.

% Simulation.
WORLD  = 10;          % world size.
dt     = 0.01;        % time step.

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

title = ['radial3D-r', num2str(ROBOTS), 'g', num2str(GROUPS), '.avi'];
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
      
    % a(i, :) -> acceleration input for robot i.
    a = [nansum(ax)' nansum(ay)' nansum(az)'];
    
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

    writeVideo(video, im2frame(print(''-RGBImage'')));
    
    if norm(v) < 0.5 && count_metric_3d(q, const) < 0.001
        break;
    end
end
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sphere plot animation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x, y, z] = sphere(30);

for i = 1:GROUPS
    aux = (i-1)*ROBOTS/GROUPS+1:i*ROBOTS/GROUPS;
    hull = convhulln(q(aux, :));
    hull = unique(hull(:)) + (i-1)*ROBOTS/GROUPS;
    [c, r] = sphereFit(q(hull, :));
    if (i > 1)
        delete(fit_handle);
    end
    
    fit_handle = surf(r*x + c(1,1), r*y + c(1,2), r*z + c(1,3), ...
                'FaceColor', pallete(i, :), 'EdgeColor', pallete(i,:), ...
                'FaceAlpha', 0.2, 'EdgeAlpha', 0.8);
            
    for j = (1:ROBOTS)
        if j >= (i-1)*ROBOTS/GROUPS+1 && j <= i*ROBOTS/GROUPS
            set(handler(j), 'EdgeAlpha', 0.0, 'FaceAlpha', 1.0);                
        else
            set(handler(j), 'EdgeAlpha', 0.0, 'FaceAlpha', 0.05);
        end        
    end
    
    for j = 1:video.FrameRate
        camorbit(camspeed*dt,0);    
        fill_lgt = camlight(fill_lgt, 'headlight');
        drawnow;
        writeVideo(video, im2frame(print(''-RGBImage'')));
    end    
end

% rotate a little more
delete(fit_handle);
for i = 1:video.FrameRate
    camorbit(camspeed*dt,0);    
    fill_lgt = camlight(fill_lgt, 'headlight');
    drawnow;
    writeVideo(video, im2frame(print(''-RGBImage'')));
end    
    
close(video);
close(simulation);