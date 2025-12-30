package com.Aura_Real.Aura_Real.Customer.Repo;

import com.Aura_Real.Aura_Real.Customer.Model.Customer;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CustomerRepo extends JpaRepository <Customer,Integer>
{
    Customer findByEmailAndPassword(String email, String password);
    boolean existsByEmail(String email);
}
