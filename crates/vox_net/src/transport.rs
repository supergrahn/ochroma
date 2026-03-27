use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::replication::NetMessage;

/// A network server that listens for client connections over TCP.
pub struct GameServer {
    listener: Option<TcpListener>,
    port: u16,
    /// The actual bound port (useful when binding to port 0).
    bound_port: Option<u16>,
}

impl GameServer {
    pub fn new(port: u16) -> Self {
        Self {
            listener: None,
            port,
            bound_port: None,
        }
    }

    /// Start listening for connections. Call within a tokio runtime.
    pub async fn start(&mut self) -> Result<(), std::io::Error> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        let local_addr = listener.local_addr()?;
        self.bound_port = Some(local_addr.port());
        println!("[ochroma-net] Server listening on port {}", local_addr.port());
        self.listener = Some(listener);
        Ok(())
    }

    /// Returns the actual bound port (after `start()` has been called).
    pub fn bound_port(&self) -> Option<u16> {
        self.bound_port
    }

    /// Accept one client connection.
    pub async fn accept(&self) -> Result<TcpStream, std::io::Error> {
        if let Some(listener) = &self.listener {
            let (stream, addr) = listener.accept().await?;
            println!("[ochroma-net] Client connected: {}", addr);
            Ok(stream)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Server not started",
            ))
        }
    }

    /// Send a length-prefixed message to a stream.
    pub async fn send(stream: &mut TcpStream, msg: &NetMessage) -> Result<(), std::io::Error> {
        let data = msg.serialize();
        let len = data.len() as u32;
        stream.write_all(&len.to_le_bytes()).await?;
        stream.write_all(&data).await?;
        Ok(())
    }

    /// Receive a length-prefixed message from a stream.
    pub async fn recv(stream: &mut TcpStream) -> Result<NetMessage, std::io::Error> {
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes).await?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        if len > 16 * 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Message too large (>16 MiB)",
            ));
        }

        let mut data = vec![0u8; len];
        stream.read_exact(&mut data).await?;

        NetMessage::deserialize(&data).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to deserialize message")
        })
    }
}

/// A network client that connects to a game server.
pub struct GameClient {
    stream: Option<TcpStream>,
}

impl GameClient {
    pub fn new() -> Self {
        Self { stream: None }
    }

    /// Connect to a server at the given address (e.g. "127.0.0.1:7777").
    pub async fn connect(&mut self, addr: &str) -> Result<(), std::io::Error> {
        let stream = TcpStream::connect(addr).await?;
        println!("[ochroma-net] Connected to {}", addr);
        self.stream = Some(stream);
        Ok(())
    }

    /// Send a message to the server.
    pub async fn send(&mut self, msg: &NetMessage) -> Result<(), std::io::Error> {
        if let Some(stream) = &mut self.stream {
            GameServer::send(stream, msg).await
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Not connected",
            ))
        }
    }

    /// Receive a message from the server.
    pub async fn recv(&mut self) -> Result<NetMessage, std::io::Error> {
        if let Some(stream) = &mut self.stream {
            GameServer::recv(stream).await
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Not connected",
            ))
        }
    }

    /// Returns true if a TCP connection has been established.
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Disconnect from the server.
    pub fn disconnect(&mut self) {
        self.stream = None;
    }
}

impl Default for GameClient {
    fn default() -> Self {
        Self::new()
    }
}
