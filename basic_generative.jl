function read_dist(fname,q,L,id1,id2,id3) 
        X = readdlm(fname)    
        @show (n_max, n_ele) = size(X)
        contact_out = zeros(L,L)

        for n in 1:n_max
            i,j,value  = map(Int64, X[n,id1]),  map(Int64, X[n,id2]), map(Float64, X[n,id3])
            contact_out[i,j] = value
            contact_out[j,i] = value
        end
        return contact_out
end

function read_contact(fname,q,L,id1,id2,id3,thresh, dist) 
        X = readdlm(fname)    
        @show (n_max, n_ele) = size(X)
        contact_out = zeros(L,L)

        for n in 1:n_max
            i,j,value  = map(Int64, X[n,id1]),  map(Int64, X[n,id2]), map(Float64, X[n,id3])
            if(value<dist)
                contact_out[i,j] = 1
                contact_out[j,i] = 1
            end
        end
        return contact_out
end

function output_artificial_couplings(q, L, mean_v, var_v, cont_mat)
    coupling = zeros(q*L, q*L)
    for i in 1:L
        for j in (i+1):L
            for a in 1:q
              	#Assume nonzero entrories are only diagonal. 
		value = mean_v*cont_mat[i,j] + var_v*randn()
               	coupling[km(i,a,q), km(j,a,q)] = value
               	coupling[km(j,a,q), km(i,a,q)] = value
            end
        end
    end
    return coupling
end

#
#Generate local field so that  \sum_a f_i(a) = 1
#    q=# of states
#    L=# of residues
#    prob=frequencies of states (\sum_aa prob[a] = 10) !!! NOT 1. and prob[a] \in Integer
#    eps=pseudo count (>0)
#    Note, frequencies of amino acids are based on the prob that is permuted at each residue positions
#    Note, 
#
function output_artificial_field(rng,q,L,prob,eps)
    #h_i(a) = log(f_i(a) + eps)
    h = zeros(q*L)
    for i in 1:L
        freq = randperm!(rng, copy(prob)) * 0.1
        for a in 1:q
		h[km(i,a,q)] = log( (1-eps)*freq[a] + eps/q )
        end
    end
    return h
end
  
function main_artificial_coupling_field(q,L,fname_cotact, fname_out)
 	@show "This method use d the ad hodc probability, assume only 4 states varialavres." 
    fout = open(fname_out, "w")
    cont = read_contact(fname_cotact, q, L, 1,2,4,0.1,6.0);
    J = output_artificial_couplings(q, L, 1.0, 0.3, cont);
    h = output_artificial_field(rng,q,L,[1 2 3 4],0.01);
    for i in 1:L
        for j in (i+1):L
            for a in 1:q
                for b in 1:q
                    println(fout, "J ", i-1, " ", j-1, " ", a-1, " ", b-1, " ", J[km(i,a,q), km(j,b,q)])
                end
            end
        end
    end
    for i in 1:L
        for a in 1:q
            println(fout,"h ", i-1, " ", a-1, " ", h[km(i,a,q)])
        end
    end
    close(fout)
end

function write_msa(X, fname)
    fout = open(fname, "w")
    N,L = size(X)
    for n in 1:N
        for i in 1:L
            print(fout, X[n,i], " ")
        end
        println(fout,"")
    end
    close(fout)
end
