
"""
	init_h_J( f1::Array{Float64,1}, q::Int64, L::Int64, Meff::Float64; withdiag=false)
	return (h,J)	
"""
function init_h_J( f1::Array{Float64,1}, q::Int64, L::Int64, Meff::Float64; withdiag=false)
	#J = rand(Float64, (q*L, q*L)) * (1.0/q^2)
	J = rand(Float64, q*L, q*L) * (1.0/q^2)
	J = J+J'	
	#h = log.(f1+0.1/Meff * ones(q*L))
	h = randn(q*L)#log.(f1+0.1/Meff * ones(q*L))
	return (h,J) 
end

"""
	E_i(i::Int64, a::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	return e_i 
"""
function E_i(i::Int64, a::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	e_i = 0.0
	for j=1:L
		if j!=i
			b = A[j]+1	
			e_i += -J[ (i-1)*q+a , (j-1)*q+b ]
		end
	end
	e_i += -h[ (i-1)*q+a ]
	return e_i 
end

"""
	dE_i(i::Int64, a1::Int64, a2::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	return de_i # = E_i(a_proposed) - E_i(a 
"""
function dE_i(i::Int64, a1::Int64, a2::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	de_i = 0.0
	for j=1:L
		if j!=i
			b = A[j]+1	
			de_i += -(J[ (i-1)*q+a2 , (j-1)*q+b ] - J[ (i-1)*q+a1 , (j-1)*q+b ])
		end
	end
	de_i += -(h[ (i-1)*q+a2 ] - h[ (i-1)*q+a1 ])
	return de_i 
end

"""
	Metropolis_Hastings(i::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	return (accepted, A) 
"""
function Metropolis_Hastings(i::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	a = A[i]
	#e_i_saved = E_i(i, a, A, J, h)
	a_proposed = rand(vcat(0:(a-1), (a+1):(q-1))) # Note entries of X and A are defined as between 0 and 20. 
	#e_i_proposed = E_i(i, Ai_proposed, A, J, h)	
	dE = dE_i(i, a+1, a_proposed+1, A, J, h)
	w = exp(-dE)
	accepted = 1 
	if(w>rand())
		A[i] = a_proposed
	else
		accepted = 0 
	end	
	return (accepted, A) 
end

"""
	Monte_Carlo_sweep(L::Int64, A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	return n_accepted
"""
function Monte_Carlo_sweep(L::Int64, A::Array{Int64,1}, J::Array{Float64,2}, h::Array{Float64,1})
	n_accepted = 0
	for l=1:L
		i = rand(1:L)	
		(accepted, A) = Metropolis_Hastings(i, A, J, h)
		n_accepted += accepted
	end
	return (n_accepted,A) 
end

function pCDk(X::Array{Int64, 2}, k_max::Int64, M::Int64, q::Int64, L::Int64, J::Array{Float64, 2}, h::Array{Float64, 1}) 
	X_after_transition = zeros(Int64, M, L)
	f1 = zeros(Float64, L*q)
	f2 = zeros(Float64, L*q, L*q)
	scale = 1.0/M	
	A = rand(0:(q-1), L)
	for m=1:M
		for k=1:k_max
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		
		for i in 1:L
			a = X[m,i]+1 #1 to 21
			f1[(i-1)*q+a] += scale
			X_after_transition[m,i] = A[i]	
			for j in (i+1):L
				b = X[m,j] + 1 
				f2[(i-1)*q+a, (j-1)*q+b] += scale
				f2[(j-1)*q+b, (i-1)*q+a] += scale
			end
		end
	end
	return (f1, f2, X_after_transition) 
end

function pCDk_minibatch(X_persistent::Array{Int64, 2}, id_set::Array{Int64, 1}, k_max::Int64, M::Int64, q::Int64, L::Int64, J::Array{Float64, 2}, h::Array{Float64, 1}) 
	X_after_transition = copy(X_persistent) 
	f1 = zeros(Float64, L*q)
	f2 = zeros(Float64, L*q, L*q)
	n_batch = size(id_set, 1)
	scale = 1.0/(n_batch)	
	
	A = rand(0:(q-1), L)
	for n=1:n_batch
		m = id_set[n]	
		A = X_persistent[m, :] 
		for k=1:k_max
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		
		for i in 1:L
			a = X_persistent[m,i]+1 #1 to 21
			f1[(i-1)*q+a] += scale
			X_after_transition[m,i] = A[i]	
			for j in (i+1):L
				b = X_persistent[m,j] + 1 
				f2[(i-1)*q+a, (j-1)*q+b] += scale
				f2[(j-1)*q+b, (i-1)*q+a] += scale
			end
		end
	end
	return (f1, f2, X_after_transition) 
end


function pCDk_weight(X::Array{Int64, 2}, k_max::Int64, M::Int64, W::Array{Float64,1}, q::Int64, L::Int64, J::Array{Float64, 2}, h::Array{Float64, 1}) 
	X_after_transition = zeros(Int64, M, L)
	f1 = zeros(Float64, L*q)
	f2 = zeros(Float64, L*q, L*q)
	M_eff = sum(W)	
	scale = 1.0/M_eff

	A = rand(0:(q-1), L)
	for m=1:M
		for k=1:k_max
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		
		for i in 1:L
			a = X[m,i]+1 #1 to 21
			f1[(i-1)*q+a] += W[m] * scale
			X_after_transition[m,i] = A[i]	
			for j in (i+1):L
				b = X[m,j] + 1 
				f2[(i-1)*q+a, (j-1)*q+b] += W[m] * scale
				f2[(j-1)*q+b, (i-1)*q+a] += W[m] * scale
			end
		end
	end
	return (f1, f2, X_after_transition) 
end

function gradient_ascent_sparse( lambda_h::Float64, lambda_J::Float64, reg_h::Float64, reg_J::Float64,  f1_1::Array{Float64,1},  f1_2::Array{Float64,1},  f2_2::Array{Float64,2},  f2_msa::Array{Float64,2}, J::Array{Float64, 2},Jfilter::Array{Float64, 2}, h::Array{Float64, 1})
	C2 = f2_2 - f1_2*f1_2'
	C_msa= f2_msa - f1_1*f1_1'
	
	dh = f1_1 - f1_2
	#dJ = (C_msa - C2 ) .* Jfilter 
	dJ = (f2_msa - f2_2 ) .* Jfilter 
	h = h * (1.0 - reg_h) + lambda_h * dh   
	J =  J  + (lambda_J * dJ - lambda_J * J ).* Jfilter  
	
	#c1vec = reshape(C_msa, (L*L*q*q))
	#c2vec = reshape(C2, (L*L*q*q))
	c1vec = zeros(Float64, Int64(L*(L-1)*q*q/2) )
	c2vec = zeros(Float64, Int64(L*(L-1)*q*q/2) )
	pos = 0
	for i = 1:L
	    # id[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= withdiag
	    for j = (i+1):L
	        # id[(j-1)*q .+ (1:q),(i-1)*q .+ (1:q)] .= true
	        c1vec[pos .+ (1:q^2)] .= vec(C_msa[(i-1)*q .+ (1:q), (j-1)*q .+ (1:q)])
	        c2vec[pos .+ (1:q^2)] .= vec(C2[(i-1)*q .+ (1:q), (j-1)*q .+ (1:q)])
	        pos += q^2
	    end
	end
	cc = Statistics.cor(c1vec,c2vec)
	cslope = linreg(c1vec,c2vec)[2]
	froc = LinearAlgebra.norm(c1vec - c2vec)

	return (J,h,
		sqrt(sum(dh.^2)), sqrt(sum(dJ.^2)), 
		cc,cslope,froc) 
end




function gradient_ascent( lambda_h::Float64, lambda_J::Float64, reg_h::Float64, reg_J::Float64,  f1_1::Array{Float64,1},  f1_2::Array{Float64,1},  f2_2::Array{Float64,2},  C_msa::Array{Float64,2}, J::Array{Float64, 2}, h::Array{Float64, 1})
	#C1 = f2_1 - f1_1*f1_1' # C1 is only possitive.
	
	C2 = f2_2 - f1_2*f1_2'
	
	dh = f1_1 - f1_2
	dJ = f2_1 - f2_2
	#dJ = C_msa - C2 
	h = h * (1.0 - reg_h) + lambda_h * dh   
	J = J * (1.0 - reg_J) + lambda_J * dJ 
	
	# The computational cost of the gradient_ascent is not significant compare with MC. 
	for i in 1:L
		for j in (1+i):L
			for a in 1:q
				for b in 1:q


				end
			end
		end
	end
	
	#c1vec = reshape(C_msa, (L*L*q*q))
	#c2vec = reshape(C2, (L*L*q*q))
	c1vec = zeros(Float64, Int64(L*(L-1)*q*q/2) )
	c2vec = zeros(Float64, Int64(L*(L-1)*q*q/2) )
	pos = 0
	for i = 1:L
	    # id[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= withdiag
	    for j = (i+1):L
	        # id[(j-1)*q .+ (1:q),(i-1)*q .+ (1:q)] .= true
	        c1vec[pos .+ (1:q^2)] .= vec(C_msa[(i-1)*q .+ (1:q), (j-1)*q .+ (1:q)])
	        c2vec[pos .+ (1:q^2)] .= vec(C2[(i-1)*q .+ (1:q), (j-1)*q .+ (1:q)])
	        pos += q^2
	    end
	end
	cc = Statistics.cor(c1vec,c2vec)
	cslope = linreg(c1vec,c2vec)[2]
	froc = LinearAlgebra.norm(c1vec - c2vec)

	return (J,h,
		sqrt(sum(dh.^2)), sqrt(sum(dJ.^2)), 
		cc,cslope,froc) 
end

function get_statistics_BM(L::Int64, q::Int64,  n_sample::Int64,  n_weight::Int64, T_eq::Int64, J::Array{Float64, 2}, h::Array{Float64, 1})
	A = rand(0:(q-1), L)	
	X_output = zeros(Int64, n_sample, L)	
	
	for m=1:T_eq
		(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
	end	
	for m=1:n_sample
		for t=1:n_weight
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		for i=1:L
			X_output[m,i] = A[i]	
		end
	end
	return X_output
end

function output_statistics(t::Int64, L::Int64, q::Int64,  n_sample::Int64, h::Array{Float64, 1}, J::Array{Float64, 2})
	fname_out = "./statistics-t-" *string(t)* ".txt"
	fout = open(fname_out, "w")
	T_eq, T_aut = 1000, 30	
	
	A = rand(0:(q-1), L)	
	for m=1:T_eq
		(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
	end	
	#X_output = zeros(Int64, n_sample, L)	
	
	for m=1:n_sample
		for t=1:T_aut
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		
		for i=1:L
			#X_output[m,i] = A[i]	
			print(fout,A[i], " ")
		end
		println(fout,"")
	end
	close(fout)
end

function output_statistics(fname_out::String, L::Int64, q::Int64,  n_sample::Int64, h::Array{Float64, 1}, J::Array{Float64, 2})
	fout = open(fname_out, "w")
	T_eq, T_aut = 1000, 30	
	
	A = rand(0:(q-1), L)	
	for m=1:T_eq
		(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
	end	
	#X_output = zeros(Int64, n_sample, L)	
	
	for m=1:n_sample
		for t=1:T_aut
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		
		for i=1:L
			#X_output[m,i] = A[i]	
			print(fout,A[i], " ")
		end
		println(fout,"")
	end
	close(fout)
end



function output_paramters(t::Int64, L::Int64, q::Int64, h::Array{Float64, 1}, J::Array{Float64,2})
	fname_out = "./parameters-t"*string(t)*".txt"
	fout = open(fname_out, "w")
	for i=1:L
		for j=(i+1):L
			for a=1:q
				for b=1:q
					println(fout, "J ", i-1, " ", j-1, " ", a-1, " ", b-1, " ", J[(i-1)*q+a, (j-1)*q+b])
				end
			end
		end
	end
	for i=1:L
		for a=1:q
			println(fout, "h ", i-1, " ", a-1, " ", h[(i-1)*q+a])
		end
	end
	
	close(fout)
end



