function [templates,templates_weights]=update_templates(curr_template,templates,n_templates_max)
[n_features,n_templates_now]=size(templates);
if n_templates_now<n_templates_max
    templates=[templates,curr_template];
    templates_weights=ones(1,size(templates,2))/size(templates,2);
else
    
    simi_t_t=compute_similarity(templates,templates);
    simi_self=sum(sum(simi_t_t))-length(simi_t_t);
    simi_self=simi_self/(numel(simi_t_t)-length(simi_t_t));
    
    simi_c_t=compute_similarity(curr_template,templates);
    simi_new=sum(simi_c_t)/length(simi_c_t);
    
    if simi_new>simi_self
        templates(:,end)=curr_template;
        disp('Successfully Update')
    end
    simi_t_t=compute_similarity(templates,templates);
    templates_weights=sum(simi_t_t)-1;
    templates_weights=templates_weights/sum(templates_weights);
    
    [templates_weights,sort_idxs]=sort(templates_weights,'descend');
    templates=templates(:,sort_idxs);
end