function count = count_metric_3d(q, const)
    robots = size(q);
    robots = robots(1);
    q = q - repmat(mean(q), robots, 1);
    dist = q(:,1).^2 + q(:,2).^2 + q(:,3).^2;
    type = diag(const);
    
    tij_less  = bsxfun(@lt, type, type');
    tij_great = bsxfun(@gt, type, type');
    dij_less  = bsxfun(@le, dist, dist');
    dij_great = bsxfun(@ge, dist, dist');
    
    eij = (tij_less & dij_great) | (tij_great & dij_less);
    count = sum(eij(:))/(robots^2);
end