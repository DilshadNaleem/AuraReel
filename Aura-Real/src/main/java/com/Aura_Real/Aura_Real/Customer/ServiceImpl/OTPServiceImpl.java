package com.Aura_Real.Aura_Real.Customer.ServiceImpl;

import com.Aura_Real.Aura_Real.Customer.Service.OTPService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Service;

import java.util.Date;

import static reactor.netty.http.HttpConnectionLiveness.log;

@Service
public class OTPServiceImpl implements OTPService {
    private static final String OTP_ATTRIBUTE = "verificationOtp";
    private static final String OTP_EMAIL_ATTRIBUTE = "verificationEmail";

    @Override
    public String generateOtp() {
        int otp = (int) (Math.random() * 900000) + 100000;
        String otpString = String.valueOf(otp);
        log.info("Generated OTP: {}", otpString);
        return otpString;
    }

    @Override
    public void storeOTP(HttpSession session, String email, String otp) {
        log.info("Storing OTP in session - Session ID: {}", session.getId());
        log.info("Before storage - verificationEmail: {}", session.getAttribute(OTP_EMAIL_ATTRIBUTE));
        log.info("Before storage - verificationOtp: {}", session.getAttribute(OTP_ATTRIBUTE));

        // Store OTP info
        session.setAttribute(OTP_EMAIL_ATTRIBUTE, email);
        session.setAttribute(OTP_ATTRIBUTE, otp);

        // Also store email with regular key for compatibility
        session.setAttribute("email", email);

        // Set session timeout to 5 minutes
        session.setMaxInactiveInterval(300);

        log.info("After storage - verificationEmail: {}", session.getAttribute(OTP_EMAIL_ATTRIBUTE));
        log.info("After storage - verificationOtp: {}", session.getAttribute(OTP_ATTRIBUTE));
        log.info("Session timeout set to: {} seconds", session.getMaxInactiveInterval());
        log.info("OTP successfully stored for email: {}", email);
    }

    @Override
    public boolean validateOTP(HttpSession session, String email, String otp) {
        log.info("Validating OTP - Session ID: {}", session.getId());

        String sessionEmail = (String) session.getAttribute(OTP_EMAIL_ATTRIBUTE);
        String sessionOtp = (String) session.getAttribute(OTP_ATTRIBUTE);

        log.info("Expected email: {}, Session email: {}", email, sessionEmail);
        log.info("Provided OTP: {}, Session OTP: {}", otp, sessionOtp);

        if (sessionEmail == null || sessionOtp == null) {
            log.error("Session attributes are null. Session might have expired.");
            log.error("Session creation time: {}", new Date(session.getCreationTime()));
            return false;
        }

        boolean isValid = email.equals(sessionEmail) && otp.equals(sessionOtp);
        log.info("OTP validation result: {}", isValid);
        return isValid;
    }
}