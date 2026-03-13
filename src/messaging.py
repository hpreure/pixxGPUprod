"""
pixxEngine Messaging Module
============================
RabbitMQ wrapper for publishing and consuming messages.
"""

import pika
import json
import logging
from typing import Optional

from src.detection_config import settings

# Configure logging
logger = logging.getLogger(__name__)


class RabbitMQClient:
    """
    RabbitMQ client wrapper for pixxEngine.
    Handles connection, publishing, and consuming messages.
    """
    
    def __init__(self, host: str = None, port: int = None, 
                 user: str = None, password: str = None, vhost: str = None):
        """
        Initialize RabbitMQ client.
        
        Args:
            host: RabbitMQ host (defaults to settings)
            port: RabbitMQ port (defaults to settings)
            user: Username (defaults to settings)
            password: Password (defaults to settings)
            vhost: Virtual host (defaults to settings)
        """
        self.host = host or settings.RABBITMQ_HOST
        self.port = port or settings.RABBITMQ_PORT
        self.user = user or settings.RABBITMQ_USER
        self.password = password or settings.RABBITMQ_PASSWORD
        self.vhost = vhost or settings.RABBITMQ_VHOST
        
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        
    def _get_credentials(self) -> pika.PlainCredentials:
        """Get Pika credentials object."""
        return pika.PlainCredentials(self.user, self.password)
    
    def _get_connection_params(self) -> pika.ConnectionParameters:
        """Get Pika connection parameters."""
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=self._get_credentials(),
            heartbeat=0,  # Heartbeat disabled — GPU inference blocks the thread for 100ms–10s.
            # Safe for local/LAN broker (192.168.x.x). Risk: if the local switch
            # restarts, Python will not detect the dead TCP connection and the
            # publisher will hang. Mitigate by setting socket_timeout below.
            blocked_connection_timeout=600,  # 10 min timeout for blocked connections
            socket_timeout=30,
        )
    
    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        if self._connection is None or self._connection.is_closed:
            logger.info(f"Connecting to RabbitMQ: {self.host}:{self.port}/{self.vhost}")
            self._connection = pika.BlockingConnection(self._get_connection_params())
            self._channel = self._connection.channel()
            logger.info("RabbitMQ connection established")
    
    def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            self._connection.close()
            logger.info("RabbitMQ connection closed")
        self._connection = None
        self._channel = None
    
    def ensure_connection(self) -> None:
        """Ensure connection is active, reconnect if needed."""
        if self._connection is None or self._connection.is_closed:
            self.connect()
        if self._channel is None or self._channel.is_closed:
            self._channel = self._connection.channel()
    
    @property
    def channel(self) -> pika.channel.Channel:
        """Get the underlying Pika channel (for advanced operations)."""
        self.ensure_connection()
        return self._channel

    def declare_queue(self, queue_name: str, durable: bool = True, 
                      arguments: dict = None) -> None:
        """
        Declare a queue (creates if doesn't exist).
        
        Args:
            queue_name: Name of the queue
            durable: Whether queue survives broker restart
            arguments: Additional queue arguments
        """
        self.ensure_connection()
        self._channel.queue_declare(
            queue=queue_name,
            durable=durable,
            arguments=arguments or {}
        )
        logger.debug(f"Queue declared: {queue_name}")
    
    def publish(self, queue: str, message: str, persistent: bool = True) -> bool:
        """
        Publish a message to a queue.
        
        Args:
            queue: Queue name
            message: Message body (string, typically JSON)
            persistent: Whether message survives broker restart
        
        Returns:
            True if published successfully
        """
        try:
            self.ensure_connection()
            self.declare_queue(queue)
            
            properties = pika.BasicProperties(
                delivery_mode=2 if persistent else 1,  # 2 = persistent
                content_type='application/json',
            )
            
            self._channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=message,
                properties=properties
            )
            
            logger.debug(f"Published message to {queue}: {message[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to {queue}: {e}")
            self.close()  # Force reconnect on next publish
            return False
    
    def publish_json(self, queue: str, data: dict, persistent: bool = True) -> bool:
        """
        Publish a JSON-serializable dict to a queue.
        
        Args:
            queue: Queue name
            data: Dictionary to serialize and publish
            persistent: Whether message survives broker restart
        
        Returns:
            True if published successfully
        """
        return self.publish(queue, json.dumps(data), persistent)


def create_mq_client() -> RabbitMQClient:
    """Factory function to create a configured RabbitMQ client."""
    return RabbitMQClient(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        user=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASSWORD,
        vhost=settings.RABBITMQ_VHOST
    )


def create_local_connection(
    heartbeat: int = 600,
    blocked_connection_timeout: int = 300,
    **kwargs,
) -> pika.BlockingConnection:
    """Return a pika BlockingConnection to the **local** RabbitMQ broker.

    All workers that need a raw pika connection should call this instead
    of inlining ConnectionParameters.
    """
    params = pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host=settings.RABBITMQ_VHOST,
        credentials=pika.PlainCredentials(
            settings.RABBITMQ_USER,
            settings.RABBITMQ_PASSWORD,
        ),
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_connection_timeout,
        **kwargs,
    )
    return pika.BlockingConnection(params)


def create_vps_connection(
    heartbeat: int = 600,
    blocked_connection_timeout: int = 300,
    **kwargs,
) -> pika.BlockingConnection:
    """Return a pika BlockingConnection to the **VPS** RabbitMQ broker.

    Used by workers that publish notifications to the remote VPS.
    """
    params = pika.ConnectionParameters(
        host=settings.VPS_RABBITMQ_HOST,
        port=settings.VPS_RABBITMQ_PORT,
        virtual_host=settings.VPS_RABBITMQ_VHOST,
        credentials=pika.PlainCredentials(
            settings.VPS_RABBITMQ_USER,
            settings.VPS_RABBITMQ_PASSWORD,
        ),
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_connection_timeout,
        **kwargs,
    )
    return pika.BlockingConnection(params)
