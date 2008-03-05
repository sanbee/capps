clear

file = fopen('dasp.dat','r','native');

mat(100)=0;

clumpcount = 0;


i=1;


while(!feof(file))

  [tok,cnt] = fscanf(file,"%s",1);

  if(strcmp(tok,"dAsp:"))
   clumpcount = clumpcount + 1;
   [tok,cnt] = fscanf(file,"%s",1);
   [tok,cnt] = fscanf(file,"%s",1);
   mat(clumpcount) = eval(tok);
   [tok,cnt] = fscanf(file,"%s",1);
  else 
    if(strcmp(tok,"###Info:"))
     
     if(clumpcount>0)
       
       hist(mat(1:clumpcount),[1:50],10);
       disp("press a key...");
       pause;
     
     end
     
     clumpcount = 0;
     [tok,cnt] = fscanf(file,"%s",1);
     [tok,cnt] = fscanf(file,"%s",1);
     [tok,cnt] = fscanf(file,"%s",1);
    end
  end

  
end

