package com.Aura_Real.Aura_Real.Customer.Service;

public interface EmailService
{
    void sendVerificationEmail (String toEmail, String otp);
}
