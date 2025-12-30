package com.Aura_Real.Aura_Real.Customer.Service;

import com.Aura_Real.Aura_Real.Customer.Model.Customer;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.ResponseEntity;

public interface CustomerService {
    Customer save(Customer customer);
    Customer findByEmailAndPassword(String email, String password);
    boolean existsByEmail(String email,int status);
    ResponseEntity<?> verifyOtp (String otp, HttpSession session);
}