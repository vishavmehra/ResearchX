-- Private document originals live in Supabase Storage (not in PostgreSQL). Objects are stored at
-- user-documents/<auth user id>/<document id>; only the owner can read, write or delete them. The backend calls the
-- Storage API with the user's own access token, so these policies (not a service key) authorise every operation.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('user-documents', 'user-documents', false, 5242880, array['application/pdf', 'text/plain', 'text/markdown'])
on conflict (id) do nothing;

create policy rx_user_documents_select on storage.objects for select to authenticated
  using (bucket_id = 'user-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy rx_user_documents_insert on storage.objects for insert to authenticated
  with check (bucket_id = 'user-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy rx_user_documents_delete on storage.objects for delete to authenticated
  using (bucket_id = 'user-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
