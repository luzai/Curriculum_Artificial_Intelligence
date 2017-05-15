function res=cmp_weights(curr,weights)
res=false;
weights=reshape(weights,1,length(weights));
all_weights=[curr,weights];
all_weights=all_weights/sum(all_weights);
if all_weights(1)>all_weights(end)
    res=true;
end

