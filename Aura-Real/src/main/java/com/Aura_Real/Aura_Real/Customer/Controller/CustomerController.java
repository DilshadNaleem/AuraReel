package com.Aura_Real.Aura_Real.Customer.Controller;

import com.Aura_Real.Aura_Real.Customer.DTO.CustomerRequest;
import com.Aura_Real.Aura_Real.Customer.Model.Customer;
import com.Aura_Real.Aura_Real.Customer.Service.CustomerService;
import com.Aura_Real.Aura_Real.Customer.Service.EmailService;
import com.Aura_Real.Aura_Real.Customer.Service.OTPService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

import static reactor.netty.http.HttpConnectionLiveness.log;

@RestController
@RequestMapping("/Customer")
@CrossOrigin(origins = "${react.api.url}", allowCredentials = "true")
public class CustomerController {

    private final CustomerService customerService;
    private final OTPService otpService;
    private final EmailService emailService;

    @Value("${react.api.url}")
    private String reactUrl;

    public CustomerController(CustomerService customerService,
                              OTPService otpService,
                              EmailService emailService) {
        this.customerService = customerService;
        this.otpService = otpService;
        this.emailService = emailService;
        log.info("CustomerController initialized with all services");
    }

    @PostMapping("/Register")
    public ResponseEntity<?> Register(@RequestBody CustomerRequest request, HttpSession session) {
        try {
            log.info("=== REGISTRATION START ===");

            if (request == null) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Request body is required");
                return ResponseEntity.badRequest().body(errorResponse);
            }

            // Validate required fields
            if (request.getEmail() == null || request.getEmail().isEmpty()) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Email is required");
                return ResponseEntity.badRequest().body(errorResponse);
            }
            if (request.getPassword() == null || request.getPassword().isEmpty()) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Password is required");
                return ResponseEntity.badRequest().body(errorResponse);
            }
            if (request.getFirstName() == null || request.getFirstName().isEmpty()) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Full name is required");
                return ResponseEntity.badRequest().body(errorResponse);
            }

            // Check if email already exists
            if (customerService.existsByEmail(request.getEmail(), 1)) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Email already registered");
                return ResponseEntity.status(HttpStatus.CONFLICT).body(errorResponse);
            }

            log.info("Session ID during registration: {}", session.getId());
            log.info("Is new session? {}", session.isNew());

            // Store ALL customer data in session
            session.setAttribute("email", request.getEmail());
            session.setAttribute("password", request.getPassword());
            session.setAttribute("firstName", request.getFirstName());
            session.setAttribute("contact", request.getContactNumber());
            session.setAttribute("nic", request.getNic());

            // CRITICAL: Also store email as verificationEmail for OTP service
            session.setAttribute("verificationEmail", request.getEmail());

            log.info("Stored in session - email: {}", session.getAttribute("email"));
            log.info("Stored in session - verificationEmail: {}", session.getAttribute("verificationEmail"));

            // Generate OTP
            String otp = otpService.generateOtp();
            log.info("Generated OTP: {}", otp);

            try {
                // Send email
                emailService.sendVerificationEmail(request.getEmail(), otp);

                // Store OTP in session
                otpService.storeOTP(session, request.getEmail(), otp);

                log.info("OTP stored successfully");
                log.info("Session verificationEmail after OTP storage: {}", session.getAttribute("verificationEmail"));
                log.info("Session verificationOtp after OTP storage: {}", session.getAttribute("verificationOtp"));

                // Create success response
                Map<String, Object> response = new HashMap<>();
                response.put("success", true);
                response.put("message", "Verify the Email");
                response.put("sessionId", session.getId()); // For debugging
                response.put("sessionCreated", !session.isNew()); // Session should not be new after setting attributes

                log.info("=== REGISTRATION END ===");
                return ResponseEntity.status(HttpStatus.CREATED).body(response);
            }
            catch (Exception e) {
                log.error("Failed to send OTP email {}", request.getEmail(), e);
                Map<String,Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Failed to Send Email " + e.getMessage());
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
            }

        } catch (Exception e) {
            log.error("Registration failed: ", e);
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Registration failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @PostMapping("/Signing")
    public ResponseEntity<?> SignIn(@RequestBody Map<String, String> credentials) {
        try {
            String email = credentials.get("email");
            String password = credentials.get("password");

            if (email == null || email.isEmpty() || password == null || password.isEmpty()) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Email and password are required");
                return ResponseEntity.badRequest().body(errorResponse);
            }

            // Authenticate user
            Customer customer = customerService.findByEmailAndPassword(email, password);

            if (customer == null) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Invalid email or password");
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
            }

            // Check if account is active
            if (customer.getStatus() != 1) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("success", false);
                errorResponse.put("message", "Account is inactive. Please contact support");
                return ResponseEntity.status(HttpStatus.FORBIDDEN).body(errorResponse);
            }

            // Create success response
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Login successful");
            response.put("customerId", customer.getId());
            response.put("email", customer.getEmail());
            response.put("name", customer.getFirstName());
            response.put("contactNumber", customer.getContactNumber());
            response.put("nic", customer.getNic());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Login failed: ", e);
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Login failed: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }
}