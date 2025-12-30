package com.Aura_Real.Aura_Real.Customer.Service;

import jakarta.servlet.http.HttpSession;

public interface OTPService
{
    String generateOtp();


    void storeOTP(HttpSession session, String email, String otp);
    boolean validateOTP (HttpSession session, String email, String otp);

}
