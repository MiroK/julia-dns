A = rand(Float64, 16, 16, 16)
B = rand(Complex{Float64}, 9, 16, 16)


fA = rfft(A, (1, 2, 3))

RFFT = plan_rfft(A, (1, 2, 3))
fAp = similar(fA)
A_mul_B!(fAp, RFFT, A)
#fAp = RFFT*A
println(sumabs2(fA-fAp))


AA = irfft(fA, 16, (1, 2, 3))

IRFFT = plan_irfft(B, 16, (1, 2, 3))
AAp = similar(AA)
A_mul_B!(AAp, IRFFT, fAp)
#AAp = IRFFT*fAp

println(sumabs2(AA-AAp))
