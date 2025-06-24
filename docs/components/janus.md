# JANUS Protocol (SDAP Addendum)

**Document ID:** MEMSHADOW-IMPL-005 (Adapted from JANUS.md)

The JANUS Protocol is an addendum to the Secure Databurst Archival Protocol (SDAP). It specifically addresses the need for **portable sealing mechanisms** for encrypting SDAP archives, ensuring that security is not rigidly tied to non-transferable hardware like a TPM. The core idea is to allow the archival and restoration process to be securely migrated across different host environments (virtual or physical).

## 1. Acknowledgment of Revised Mandate

-   **Original Goal:** Hardware-binding (e.g., TPM-based) for maximum security on a specific host.
-   **Revised Goal:** Operational portability and simplified disaster recovery. The security of the archival module (specifically its encryption keys) must be transferable by an authorized operator.
-   **Key Principle:** The recompilation or secure reconfiguration workflow by an authorized operator on a new host is a primary security control. An adversary stealing only a compiled binary or an encrypted archive should not be able to decrypt data.

## 2. Portable Sealing Mechanisms

"Portable seal" refers to a secret or mechanism intrinsically tied to the encryption process or its operational environment, which is highly secure and can be reconstituted on a new host by an authorized operator. For SDAP, this relates to how the `GPG_PASSPHRASE` (used in `sdap_backup.sh`) is protected and managed.

### Option 1: "Two-Man Rule" - Split-Key Obfuscation (Less Relevant for SDAP's GPG)

This approach is more suited for custom-compiled binaries where key parts are injected at compile time.
-   **Concept:** Master encryption key split into parts. One part in source code, another injected at compile time.
-   **SDAP Relevance:** Less direct for `sdap_backup.sh` which uses GPG with a passphrase. Could apply if a custom tool managed the GPG passphrase, splitting its generation.

### Option 2: "Smart Key" - Environment-Derived Sealing (Recommended for GPG Passphrase Management)

This is the **recommended approach** for managing the `GPG_PASSPHRASE` if it's not directly stored in `sdap.env`. It involves deriving the key/passphrase from stable, operator-defined environmental factors.

*   **Concept:** The system (or a script preparing the environment for `sdap_backup.sh`) does not store the GPG passphrase directly in plain text if maximum security is desired. Instead, it contains logic to read specific environmental "factors" and derive the GPG passphrase from them.
*   **Sealing Mechanism:**
    1.  **Factor Selection:** On deploying to a new VPS, the operator selects or recreates a combination of stable host identifiers or secrets.
    2.  **Recommended Factors:**
        *   **Primary Factor (Portable Secret):** The content of a specific, operator-created file (e.g., `/etc/memshadow/host_secret.key`). This file would contain a high-entropy UUID or random string, managed by the operator.
        *   **Secondary Factors (Host-Specific, Optional):** Stable system identifiers like `/etc/machine-id` (on systemd Linux) or the MAC address of the primary network interface. These add a layer of host-binding but reduce portability if they change.
    3.  **Key Derivation Function (KDF):** At runtime (before `sdap_backup.sh` needs `GPG_PASSPHRASE`), a script or process would:
        *   Read the content of these factors.
        *   Concatenate them into a single string.
        *   Use a KDF (e.g., HKDF with `openssl kdf`, or `argon2` if passphrase characteristics allow) to derive the actual `GPG_PASSPHRASE`.
        *   This derived passphrase is then made available to `sdap_backup.sh` (e.g., by populating `sdap.env` dynamically or exporting the variable).

    ```c
    // Pseudocode for conceptual derivation logic (if implemented in C for a key manager)
    // char* factor1 = read_file_content("/etc/memshadow/host_secret.key");
    // char* factor2 = read_system_identifier("/etc/machine-id");
    // char combined_factors[...];
    // sprintf(combined_factors, "%s:%s", factor1, factor2); // Concatenate
    //
    // // Use HKDF to derive the final key/passphrase from the combined environmental factors
    // // HKDF(output_passphrase, salt="JANUS_ENV_SEAL", input_material=combined_factors, info="GPGPassphraseForSDAP");
    // free(factor1); free(factor2);
    ```
    In practice, for `sdap_backup.sh`, this might look like a preliminary script:
    ```bash
    # Example: derive_gpg_passphrase.sh
    HOST_SECRET=$(cat /etc/memshadow/host_secret.key)
    MACHINE_ID=$(cat /etc/machine-id || echo "default_machine_id") # Fallback if no machine-id
    # A more robust KDF tool or script would be used here.
    # This is a simplified example and NOT cryptographically secure for production.
    DERIVED_PASSPHRASE=$(echo "${HOST_SECRET}:${MACHINE_ID}" | sha256sum | awk '{print $1}')
    echo "GPG_PASSPHRASE=${DERIVED_PASSPHRASE}" > /etc/memshadow/sdap.env
    chmod 600 /etc/memshadow/sdap.env
    ```
    **Important:** The example above using `sha256sum` is **not a secure KDF** for passphrases. A proper KDF like Argon2 or HKDF (via `openssl` or a dedicated utility) should be used. The `sdap.env` approach is simpler and relies on filesystem permissions for security.

*   **Workflow & Portability:**
    *   The `sdap_backup.sh` script remains the same.
    *   To migrate:
        1.  Set up the new VPS.
        2.  Securely transfer or recreate the `/etc/memshadow/host_secret.key` file.
        3.  If other system identifiers are used, ensure they are noted (though primary reliance on the `host_secret.key` is better for portability).
        4.  Run the derivation script (if used) to generate `sdap.env`, or directly place the `sdap.env` file with the correct `GPG_PASSPHRASE`.
    *   The `sdap_backup.sh` script will then use the available `GPG_PASSPHRASE` for encryption.

*   **Analysis:**
    *   **Pros:** Excellent security and operational flexibility. The core secret (`host_secret.key`) is portable. An adversary needs both the backup script logic and the `host_secret.key` (and potentially other environmental factors if the KDF is complex).
    *   **Cons:** Stability of chosen environmental factors is key if they are used beyond the main operator-controlled secret file. Complexity in securely implementing the KDF if a derivation script is used.

### Option 3: "Network Seal" - Remote Key Fetching (More Complex)

This externalizes key management to a dedicated "Key Oracle" service.
-   **Concept:** The JANUS binary (or a script before `sdap_backup.sh`) fetches the GPG passphrase from a secure, isolated microservice at runtime.
-   **Mechanism:** Requires mTLS authentication between the VPS and the Key Oracle.
-   **SDAP Relevance:** Could be used if `sdap_backup.sh` was modified to fetch `GPG_PASSPHRASE` from such a service.
-   **Analysis:**
    *   **Pros:** Centralized key management, enables seamless key rotation.
    *   **Cons:** Introduces additional infrastructure (Key Oracle), single point of failure, network dependency.

## 3. Recommendation for SDAP

For the current SDAP implementation using `sdap_backup.sh` and GPG:

1.  **Baseline:** Securely storing `GPG_PASSPHRASE` in `/etc/memshadow/sdap.env` with strict root-only permissions (`chmod 600 /etc/memshadow/sdap.env`). The portability of this file (or its content) is paramount for recovery. This is the simplest approach.
2.  **Enhanced (aligning with JANUS Option 2):** If a higher level of security for the passphrase at rest is desired, implement a pre-execution script that derives `GPG_PASSPHRASE` using **Environment-Derived Sealing**.
    *   The primary secret would be an operator-managed file (e.g., `/etc/memshadow/host_secret.key`).
    *   A script uses this file (and optionally other stable system identifiers) with a strong KDF to generate the `GPG_PASSPHRASE`.
    *   This derived passphrase is then written to `/etc/memshadow/sdap.env` just before `sdap-backup.service` runs, or exported as an environment variable directly for the `ExecStart` command if systemd allows dynamic environment variable setting from script output.

The choice depends on the desired balance between operational simplicity and the security posture for the GPG passphrase at rest on the VPS. The key takeaway from JANUS is to ensure that the method chosen is **portable** and allows an authorized operator to reconstitute the encryption capability on a new system.
