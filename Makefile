.PHONY: start stop restart logs clean db-shell psql mysql redis ollama

start:
	@./start.sh

stop:
	@./stop.sh

restart: stop start

logs:
	@docker-compose logs -f

clean:
	@docker-compose down -v
	@docker system prune -f

db-shell:
	@docker exec -it heroes_postgres psql -U heroes_user -d heroes_db

psql:
	@docker exec -it heroes_postgres psql -U heroes_user -d heroes_db

mysql:
	@docker exec -it heroes_mysql mysql -u heroes_user -pheroes_password heroes_db

redis:
	@docker exec -it heroes_redis redis-cli

ollama:
	@docker exec -it heroes_ollama ollama list

status:
	@docker-compose ps

backup:
	@mkdir -p backups
	@docker exec heroes_postgres pg_dump -U heroes_user heroes_db > backups/heroes_backup_$$(date +%Y%m%d_%H%M%S).sql

update-models:
	@docker exec heroes_ollama /scripts/download-models.sh