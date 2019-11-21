ROBOTS = 150;

tasks = {%@segregation_video, ROBOTS, 5;
	 %@segregation_video, ROBOTS, 10;
	 %@segregation_video, ROBOTS, 15;
         %@segregation3d_video, ROBOTS, 5;
	 %@segregation3d_video, ROBOTS, 10;
	 %@segregation3d_video, ROBOTS, 15;
         
         @segregation_radial_videos, ROBOTS, 5;
	 @segregation_radial_videos, ROBOTS, 10;
	 @segregation_radial_videos, ROBOTS, 15;
	 @segregation_radial3d_videos, ROBOTS, 5;
         @segregation_radial3d_videos, ROBOTS, 10;
         @segregation_radial3d_videos, ROBOTS, 15;
        };
     
for i = 1:size(tasks, 1)
    func_handle = cell2mat(tasks(i, 1));
    robots = cell2mat(tasks(i, 2));
    groups = cell2mat(tasks(i, 3));
    feval(func_handle, robots, groups);
end