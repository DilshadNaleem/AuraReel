package com.Aura_Real.Aura_Real.Customer.Controller;

import com.Aura_Real.Aura_Real.Customer.Model.Customer;
import com.Aura_Real.Aura_Real.Customer.Repo.CustomerRepo;
import com.Aura_Real.Aura_Real.Customer.Service.CustomerService;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

import static reactor.netty.http.HttpConnectionLiveness.log;

@RestController
@RequestMapping("/Customer")
public class OTPVerification {

    private final CustomerService customerService;
    private final CustomerRepo customerRepo;

    public OTPVerification(CustomerService customerService, CustomerRepo customerRepo) {
        this.customerService = customerService;
        this.customerRepo = customerRepo;
        log.info("OTPVerification controller initialized");
    }

    @PostMapping("/verification")
    public ResponseEntity<?> verifyOtp(@RequestBody Map<String,String> requestBody, HttpSession session) {
        try {
            log.info("=== OTP VERIFICATION CONTROLLER START ===");

            String otp = requestBody.get("otp");
            log.info("Received OTP from request: {}", otp);
            log.info("Session ID in controller: {}", session.getId());
            log.info("Is new session in controller? {}", session.isNew());

            // Debug: List all session attributes
            log.info("Session attributes in controller:");
            java.util.Enumeration<String> attributeNames = session.getAttributeNames();
            while (attributeNames.hasMoreElements()) {
                String name = attributeNames.nextElement();
                log.info("  {} = {}", name, session.getAttribute(name));
            }

            // First verify OTP
            log.info("Calling customerService.verifyOtp()...");
            ResponseEntity<?> otpVerificationResult = customerService.verifyOtp(otp, session);

            log.info("OTP verification result status: {}", otpVerificationResult.getStatusCode());

            // Check if OTP verification failed
            Map<String, Object> responseBody = (Map<String, Object>) otpVerificationResult.getBody();
            if (responseBody == null) {
                log.error("Response body is null");
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Internal server error");
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
            }

            Boolean success = (Boolean) responseBody.get("success");
            if (success == null || !success) {
                log.error("OTP verification failed: {}", responseBody.get("message"));
                return otpVerificationResult; // Return the error from verifyOtp
            }

            log.info("OTP verified successfully, creating customer...");

            // Get email from session
            String email = (String) session.getAttribute("verificationEmail");
            if (email == null) {
                email = (String) session.getAttribute("email");
            }

            if (email == null) {
                log.error("Email not found in session");
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Email not found in session");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }

            // Create and save customer
            Customer customer = new Customer();
            customer.setEmail(email);
            customer.setPassword((String) session.getAttribute("password"));
            customer.setFirstName((String) session.getAttribute("firstName"));
            customer.setNic((String) session.getAttribute("nic"));
            customer.setContactNumber((String) session.getAttribute("contact"));
            customer.setStatus(1);

            log.info("Creating customer with email: {}", customer.getEmail());
            log.info("Customer first name: {}", customer.getFirstName());

            // Validate required fields
            if (customer.getEmail() == null || customer.getPassword() == null ||
                    customer.getFirstName() == null) {
                log.error("Required fields are missing for customer");
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Required fields are missing");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }

            Customer savedCustomer = customerRepo.save(customer);

            if (savedCustomer != null && savedCustomer.toString() != null) {
                log.info("Customer saved successfully with ID: {}", savedCustomer.getId());

                // Clear session after successful verification
                session.invalidate();

                Map<String, Object> response = new HashMap<>();
                response.put("success", true);
                response.put("message", "Account Verified. Please Login");
                response.put("customerId", savedCustomer.getId());

                log.info("=== OTP VERIFICATION CONTROLLER END - SUCCESS ===");
                return ResponseEntity.status(HttpStatus.CREATED).body(response);
            } else {
                log.error("Failed to save customer");
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Account Verification Failed - Could not save customer");
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
            }

        } catch (Exception e) {
            log.error("OTP Verification Failed: ", e);
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "OTP Verification Failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }
}