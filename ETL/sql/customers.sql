DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
customer_id VARCHAR(32) NOT NULL,
customer_unique_id VARCHAR(32) NOT NULL,
customer_zip_code_prefix INT NOT NULL,
customer_city VARCHAR(32) NOT NULL,
customer_state CHAR(2) NOT NULL);
