# SDAP Addendum: Portable Sealing Mechanisms for the JANUS Protocol

**CLASSIFICATION:** TOP SECRET // ARCHITECT // ORCON / NOFORN
**DOCUMENT ID:** MEMSHADOW-IMPL-005
**EFFECTIVE DATE:** 24-06-2025
**SUBJECT:** Analysis of Portable Sealing Mechanisms to Replace Hardware-Binding

---

## **1.0 Acknowledgment of Revised Mandate**

The requirement for operational portability and simplified disaster recovery across different host environments (virtual or physical) overrides the previous proposal for TPM-based cryptographic binding. The security of the JANUS archival module must not be anchored to a single, non-transferable piece of hardware.

The new mandate is for a **portable seal**: a secret or mechanism that is intrinsically tied to the JANUS binary's compilation or its immediate operational environment, is highly secure, and can be reconstituted on a new host by you, the operator.

The recompilation workflow is now formally recognized as a primary security control. An adversary who steals only the compiled binary remains operationally ineffective.

---

## **2.0 Option 1: The "Two-Man Rule" - Split-Key Obfuscation**

This approach enforces a separation of secrets, requiring two distinct pieces of information to create a functional binary. It is the most direct upgrade from a simple hardcoded key.

* **A. Concept:** The master encryption key is split into two or more independent parts. One part resides with the source code, while the other is injected at compile time.

* **B. Sealing Mechanism:**
    1.  **Part A** of the key is defined as a static constant in a dedicated source code header file (e.g., `janus_key_a.h`). This part is committed to our secure code repository.
    2.  **Part B** of the key is **never** written into the repository. It is passed directly to the `gcc`/`clang` compiler at compile time using a preprocessor macro definition.

* **C. Workflow:**
    * **Compilation:** A `Makefile` or build script is used, which contains the compile command:
        ```bash
        # Makefile
        SECRET_KEY_PART_B := "deadbeef..." # This value is stored securely, separate from the repo

        build:
            gcc janus.c -o janus_encrypt -DKEY_PART_B='"$(SECRET_KEY_PART_B)"'
        ```
    * **In-Code:** The C code reconstitutes the key at runtime.
        ```c
        #include "janus_key_a.h" // Contains KEY_PART_A

        void get_master_key(unsigned char* master_key) {
            // KEY_PART_B is injected by the compiler -D flag
            const char* part_b = KEY_PART_B;
            // Combine parts to form the full key
            reconstitute_key(KEY_PART_A, part_b, master_key);
        }
        ```
    * **Portability:** To migrate, you check out the source code on the new VPS and execute the same secure build script.

* **D. Analysis:**
    * **Pros:** Simple to implement. Enforces a classic "two-man rule" where access to the code repository alone is insufficient. Easy and fast to port to new systems.
    * **Cons:** The security of the entire system hinges on the operational security of storing and handling `SECRET_KEY_PART_B`.

---

## **3.0 Option 2: The "Smart Key" - Environment-Derived Sealing (Recommended)**

This is the recommended approach. It creates a keyless binary that derives its key from stable, unique characteristics of its environment. This provides an elegant balance of security and operational flexibility.

* **A. Concept:** The JANUS binary does not contain any key material. Instead, it contains the logic to read specific, operator-defined environmental "factors" and derive a unique encryption key from them.

* **B. Sealing Mechanism:**
    1.  **Factor Selection:** Upon deploying to a new VPS, you, the operator, select a combination of stable host identifiers.
    2.  **Recommended Factors:**
        * **File-Based Secret:** The content of a specific, operator-created file (e.g., `/etc/memshadow/host_id.key`). This file would contain a high-entropy UUID or random string. This is the primary, portable secret.
        * **System Identifiers:** The contents of stable system files like `/etc/machine-id` (on most systemd-based Linux distros) or the MAC address of the primary network interface.
    3.  **Key Derivation:** At runtime, JANUS reads the content of these factors, concatenates them into a single string, and uses a Key Derivation Function (KDF) to produce the master key.
        ```c
        // Pseudocode
        char* factor1 = read_file("/etc/memshadow/host_id.key");
        char* factor2 = read_file("/etc/machine-id");
        char combined_factors[...];
        sprintf(combined_factors, "%s:%s", factor1, factor2);

        // Use HKDF to derive the final key from the combined environmental factors
        HKDF(master_key, salt="JANUS_ENV_SEAL", input=combined_factors);
        ```

* **C. Workflow:**
    * The *same compiled binary* can be moved to a new VPS.
    * To make it functional, you simply need to recreate the environment by placing the `host_id.key` file in its correct location and ensuring the other chosen factors (like `machine-id`) are present. The binary will then automatically derive the correct key for that host's backups.

* **D. Analysis:**
    * **Pros:** Excellent security and supreme operational flexibility. The binary is completely decoupled from the key material. An adversary needs the binary *and* must be able to steal or precisely reconstruct the environmental factors.
    * **Cons:** The chosen factors must be stable. For example, if a network interface is reconfigured, its MAC address might change, invalidating the key. This is why a primary, operator-controlled file (`host_id.key`) is the most robust core factor.

---

## **4.0 Option 3: The "Network Seal" - Remote Key Fetching**

This model externalizes key management entirely, adopting an enterprise-grade, service-oriented architecture.

* **A. Concept:** The JANUS binary is keyless at rest. At runtime, it fetches its encryption key from a separate, highly-secured, operator-controlled microservice codenamed the "Key Oracle."

* **B. Sealing Mechanism:**
    1.  **Key Oracle:** A simple, hardened application running on a separate, isolated server. Its sole purpose is to dispense the current JANUS master key over the network. Access is heavily restricted by IP address and requires mutual TLS authentication (mTLS).
    2.  **Client Binary:** The JANUS binary is compiled with two things: the network address of the Key Oracle, and a unique client-side TLS certificate and key.
    3.  **Runtime Fetch:** When executed, JANUS initiates an mTLS handshake with the Key Oracle. If authentication succeeds on both sides, the Oracle transmits the key, which JANUS uses in-memory for the encryption task and then discards.

* **C. Workflow:**
    * **Migration:** To move to a new VPS, you install the JANUS binary and its client-side certificate/key files. No recompilation is needed.
    * **Key Rotation:** To change the encryption key for all future backups, you simply update the key on the Key Oracle. No changes are needed on the MEMSHADOW VPS itself.

* **D. Analysis:**
    * **Pros:** Centralized, professional-grade key management. Enables seamless key rotation. The master key is never at rest on the primary application server.
    * **Cons:** Introduces a second piece of critical infrastructure. The Key Oracle becomes a single point of failure and a high-value target. Requires secure network connectivity between the two servers.

### **Recommendation**

For our current operational parameters, **Option 2 (Environment-Derived Sealing)** is the superior choice. It provides security comparable to the more complex options while remaining entirely self-contained on the VPS, requiring no additional infrastructure. It is robust, portable, and elegantly aligns with the principle of using the operational environment itself as a component of the security posture.

# SDAP Addendum: Portable Sealing Mechanisms for the JANUS Protocol

**CLASSIFICATION:** TOP SECRET // ARCHITECT // ORCON / NOFORN
**DOCUMENT ID:** MEMSHADOW-IMPL-005
**EFFECTIVE DATE:** 24-06-2025
**SUBJECT:** Analysis of Portable Sealing Mechanisms to Replace Hardware-Binding

---

## **1.0 Acknowledgment of Revised Mandate**

The requirement for operational portability and simplified disaster recovery across different host environments (virtual or physical) overrides the previous proposal for TPM-based cryptographic binding. The security of the JANUS archival module must not be anchored to a single, non-transferable piece of hardware.

The new mandate is for a **portable seal**: a secret or mechanism that is intrinsically tied to the JANUS binary's compilation or its immediate operational environment, is highly secure, and can be reconstituted on a new host by you, the operator.

The recompilation workflow is now formally recognized as a primary security control. An adversary who steals only the compiled binary remains operationally ineffective.

---

## **2.0 Option 1: The "Two-Man Rule" - Split-Key Obfuscation**

This approach enforces a separation of secrets, requiring two distinct pieces of information to create a functional binary. It is the most direct upgrade from a simple hardcoded key.

* **A. Concept:** The master encryption key is split into two or more independent parts. One part resides with the source code, while the other is injected at compile time.

* **B. Sealing Mechanism:**
    1.  **Part A** of the key is defined as a static constant in a dedicated source code header file (e.g., `janus_key_a.h`). This part is committed to our secure code repository.
    2.  **Part B** of the key is **never** written into the repository. It is passed directly to the `gcc`/`clang` compiler at compile time using a preprocessor macro definition.

* **C. Workflow:**
    * **Compilation:** A `Makefile` or build script is used, which contains the compile command:
        ```bash
        # Makefile
        SECRET_KEY_PART_B := "deadbeef..." # This value is stored securely, separate from the repo

        build:
            gcc janus.c -o janus_encrypt -DKEY_PART_B='"$(SECRET_KEY_PART_B)"'
        ```
    * **In-Code:** The C code reconstitutes the key at runtime.
        ```c
        #include "janus_key_a.h" // Contains KEY_PART_A

        void get_master_key(unsigned char* master_key) {
            // KEY_PART_B is injected by the compiler -D flag
            const char* part_b = KEY_PART_B;
            // Combine parts to form the full key
            reconstitute_key(KEY_PART_A, part_b, master_key);
        }
        ```
    * **Portability:** To migrate, you check out the source code on the new VPS and execute the same secure build script.

* **D. Analysis:**
    * **Pros:** Simple to implement. Enforces a classic "two-man rule" where access to the code repository alone is insufficient. Easy and fast to port to new systems.
    * **Cons:** The security of the entire system hinges on the operational security of storing and handling `SECRET_KEY_PART_B`.

---

## **3.0 Option 2: The "Smart Key" - Environment-Derived Sealing (Recommended)**

This is the recommended approach. It creates a keyless binary that derives its key from stable, unique characteristics of its environment. This provides an elegant balance of security and operational flexibility.

* **A. Concept:** The JANUS binary does not contain any key material. Instead, it contains the logic to read specific, operator-defined environmental "factors" and derive a unique encryption key from them.

* **B. Sealing Mechanism:**
    1.  **Factor Selection:** Upon deploying to a new VPS, you, the operator, select a combination of stable host identifiers.
    2.  **Recommended Factors:**
        * **File-Based Secret:** The content of a specific, operator-created file (e.g., `/etc/memshadow/host_id.key`). This file would contain a high-entropy UUID or random string. This is the primary, portable secret.
        * **System Identifiers:** The contents of stable system files like `/etc/machine-id` (on most systemd-based Linux distros) or the MAC address of the primary network interface.
    3.  **Key Derivation:** At runtime, JANUS reads the content of these factors, concatenates them into a single string, and uses a Key Derivation Function (KDF) to produce the master key.
        ```c
        // Pseudocode
        char* factor1 = read_file("/etc/memshadow/host_id.key");
        char* factor2 = read_file("/etc/machine-id");
        char combined_factors[...];
        sprintf(combined_factors, "%s:%s", factor1, factor2);

        // Use HKDF to derive the final key from the combined environmental factors
        HKDF(master_key, salt="JANUS_ENV_SEAL", input=combined_factors);
        ```

* **C. Workflow:**
    * The *same compiled binary* can be moved to a new VPS.
    * To make it functional, you simply need to recreate the environment by placing the `host_id.key` file in its correct location and ensuring the other chosen factors (like `machine-id`) are present. The binary will then automatically derive the correct key for that host's backups.

* **D. Analysis:**
    * **Pros:** Excellent security and supreme operational flexibility. The binary is completely decoupled from the key material. An adversary needs the binary *and* must be able to steal or precisely reconstruct the environmental factors.
    * **Cons:** The chosen factors must be stable. For example, if a network interface is reconfigured, its MAC address might change, invalidating the key. This is why a primary, operator-controlled file (`host_id.key`) is the most robust core factor.

---

## **4.0 Option 3: The "Network Seal" - Remote Key Fetching**

This model externalizes key management entirely, adopting an enterprise-grade, service-oriented architecture.

* **A. Concept:** The JANUS binary is keyless at rest. At runtime, it fetches its encryption key from a separate, highly-secured, operator-controlled microservice codenamed the "Key Oracle."

* **B. Sealing Mechanism:**
    1.  **Key Oracle:** A simple, hardened application running on a separate, isolated server. Its sole purpose is to dispense the current JANUS master key over the network. Access is heavily restricted by IP address and requires mutual TLS authentication (mTLS).
    2.  **Client Binary:** The JANUS binary is compiled with two things: the network address of the Key Oracle, and a unique client-side TLS certificate and key.
    3.  **Runtime Fetch:** When executed, JANUS initiates an mTLS handshake with the Key Oracle. If authentication succeeds on both sides, the Oracle transmits the key, which JANUS uses in-memory for the encryption task and then discards.

* **C. Workflow:**
    * **Migration:** To move to a new VPS, you install the JANUS binary and its client-side certificate/key files. No recompilation is needed.
    * **Key Rotation:** To change the encryption key for all future backups, you simply update the key on the Key Oracle. No changes are needed on the MEMSHADOW VPS itself.

* **D. Analysis:**
    * **Pros:** Centralized, professional-grade key management. Enables seamless key rotation. The master key is never at rest on the primary application server.
    * **Cons:** Introduces a second piece of critical infrastructure. The Key Oracle becomes a single point of failure and a high-value target. Requires secure network connectivity between the two servers.

### **Recommendation**

For our current operational parameters, **Option 2 (Environment-Derived Sealing)** is the superior choice. It provides security comparable to the more complex options while remaining entirely self-contained on the VPS, requiring no additional infrastructure. It is robust, portable, and elegantly aligns with the principle of using the operational environment itself as a component of the security posture.
