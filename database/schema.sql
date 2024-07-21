-- database/schema.sql

-- Connect to your database
\c phdprojectdatabase;

-- Create a table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create another table with a foreign key
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial data into the users table
INSERT INTO users (username, email) VALUES ('john_doe', 'john@example.com');
INSERT INTO users (username, email) VALUES ('jane_smith', 'jane@example.com');

-- Insert initial data into the posts table
INSERT INTO posts (user_id, title, content) VALUES (1, 'First Post', 'This is the first post.');
INSERT INTO posts (user_id, title, content) VALUES (2, 'Second Post', 'This is the second post.');

-- Create an index on the username column
CREATE INDEX idx_username ON users(username);

-- Add a foreign key constraint to the posts table
ALTER TABLE posts ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id);

-- run this command:
-- psql -d phdprojectdatabase -f schema.sql
