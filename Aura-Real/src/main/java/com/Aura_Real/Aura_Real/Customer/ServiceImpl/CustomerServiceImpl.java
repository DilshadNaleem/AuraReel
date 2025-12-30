package com.Aura_Real.Aura_Real.Customer.ServiceImpl;

import com.Aura_Real.Aura_Real.Customer.Model.Customer;
import com.Aura_Real.Aura_Real.Customer.Repo.CustomerRepo;
import com.Aura_Real.Aura_Real.Customer.Service.CustomerService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import static reactor.netty.http.HttpConnectionLiveness.log;

@Service
public class CustomerServiceImpl implements CustomerService {

    @Autowired
    private CustomerRepo customerRepository;

    @Override
    public Customer save(Customer customer) {
        log.info("Saving customer with email: {}", customer.getEmail());
        return customerRepository.save(customer);
    }

    @Override
    public Customer findByEmailAndPassword(String email, String password) {
        log.info("Finding customer by email: {}", email);
        return customerRepository.findByEmailAndPassword(email, password);
    }

    @Override
    public boolean existsByEmail(String email, int status) {
        log.info("Checking if email exists: {}", email);
        return customerRepository.existsByEmail(email);
    }

    @Override
    public ResponseEntity<?> verifyOtp(String otp, HttpSession session) {
        log.info("=== OTP VERIFICATION START ===");
        log.info("Received OTP: {}", otp);
        log.info("Session ID in verifyOtp: {}", session.getId());
        log.info("Session Creation Time: {}", new Date(session.getCreationTime()));
        log.info("Session Last Accessed: {}", new Date(session.getLastAccessedTime()));
        log.info("Is New Session: {}", session.isNew());

        // List all session attributes
        log.info("All Session Attributes:");
        java.util.Enumeration<String> attributeNames = session.getAttributeNames();
        while (attributeNames.hasMoreElements()) {
            String name = attributeNames.nextElement();
            log.info("  {} = {}", name, session.getAttribute(name));
        }

        String storedOtp = (String) session.getAttribute("verificationOtp");
        String email = (String) session.getAttribute("verificationEmail");

        log.info("Customer Service Impl: Stored OTP = {}, Email = {}", storedOtp, email);

        if (email == null || storedOtp == null) {
            log.error("Email or OTP is null in session. Session may have expired.");
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Session expired. Please try again.");
            errorResponse.put("sessionId", session.getId());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }

        log.info("Comparing OTP: Stored='{}', Provided='{}'", storedOtp, otp);

        if (!storedOtp.equals(otp)) {
            log.error("OTP mismatch");
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Invalid OTP");
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }

        // OTP verified successfully
        log.info("OTP verified successfully for email: {}", email);
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("message", "OTP verified successfully");
        response.put("email", email);

        log.info("=== OTP VERIFICATION END ===");
        return ResponseEntity.ok(response);
    }
}