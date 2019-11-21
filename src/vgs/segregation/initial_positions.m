function q = initial_positions(ROBOTS, DIMENSION, WORLD, min_inter_dist)
  q = WORLD * rand(ROBOTS, DIMENSION);
  for k = 1:100
      for i = 1:ROBOTS-1
          for j = i+1:ROBOTS
              aux = q(i,:) - q(j,:);
              dist = dot(aux, aux);
              if dist < min_inter_dist * min_inter_dist
                  dist = sqrt(dist);
                  aux = aux/dist;
                  q(i,:) =  q(i,:) + 0.5*(min_inter_dist - dist)*aux;
                  q(j,:) =  q(j,:) - 0.5*(min_inter_dist - dist)*aux;
                  q(i,:) = max(min(q(i,:), WORLD), 0);
                  q(j,:) = max(min(q(j,:), WORLD), 0);
              end
          end
      end
  end
  assert(all(q(:, 1) >= 0) && all(q(:,1) <= WORLD))
  assert(all(q(:, 2) >= 0) && all(q(:,2) <= WORLD))
end
