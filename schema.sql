-- The existing users table may remain as-is; these columns must exist as described by the user.
-- login_history records successful logins for audit/reporting.
CREATE TABLE IF NOT EXISTS login_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    login_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(255) NULL,
    user_agent VARCHAR(1000) NULL,
    INDEX idx_login_history_user_time (user_id, login_at),
    CONSTRAINT fk_login_history_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
