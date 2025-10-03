    def build_visual_relationships(self, df_group, authors):
        """
        Build tables showing relationships between first words and authors in a WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for a specific group with 'message_cleaned' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            pandas.DataFrame or None: Numerical DataFrame for the second table (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted descending), or None if no data.
        """
        MIN_TOTAL = 10
        MIN_HIGHEST = 70  # in percent

        if df_group.empty:
            logger.error("Empty DataFrame provided for building visual relationships.")
            return None

        try:
            # Extract first_word from message_cleaned
            def get_first_word(message):
                if isinstance(message, str) and message.strip():
                    parts = message.split()
                    if parts:
                        return parts[0]
                return None

            df_group['first_word'] = df_group['message_cleaned'].apply(get_first_word)
            df_group = df_group.dropna(subset=['first_word'])

            if df_group.empty:
                logger.info("No valid first words found after extraction.")
                return None

            # Get counts per first_word and author
            counts = df_group.groupby(['first_word', 'author']).size().reset_index(name='count')
            pivot = counts.pivot(index='first_word', columns='author', values='count').fillna(0)

            # Calculate total
            pivot['total'] = pivot.sum(axis=1)

            # Filter out rows where total == 0 (though unlikely)
            pivot = pivot[pivot['total'] > 0]

            # Authors columns
            authors = sorted([a for a in authors if a in pivot.columns])
            if not authors:
                logger.error("No matching authors found in pivot table.")
                return None

            # Calculate percentages (numerical)
            percentages = pivot[authors].div(pivot['total'], axis=0) * 100
            percentages['highest'] = percentages.max(axis=1)

            # Combine into full numerical table: total + percentages (authors + highest)
            full_table_num = pd.concat([pivot[['total']], percentages], axis=1)

            # Create string version for logging
            full_table_str = full_table_num.copy()
            for col in authors + ['highest']:
                full_table_str[col] = full_table_str[col].apply(lambda x: f"{int(x)}%")

            # Table 1: total >= MIN_TOTAL, sorted by total descending
            table1_num = full_table_num[full_table_num['total'] >= MIN_TOTAL].sort_values('total', ascending=False)
            if not table1_num.empty:
                table1_str = full_table_str.loc[table1_num.index]
                logger.info(f"Table 1 (total >= {MIN_TOTAL}, sorted by total desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table1_str.to_string()}")
            else:
                logger.info(f"No first words with total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")

            # Table 2: highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted by highest descending
            table2_num = full_table_num[(full_table_num['highest'] >= MIN_HIGHEST) & (full_table_num['total'] >= MIN_TOTAL)].sort_values('highest', ascending=False)
            if not table2_num.empty:
                table2_str = full_table_str.loc[table2_num.index]
                logger.info(f"Table 2 (highest >= {MIN_HIGHEST}% and total >= {MIN_TOTAL}, sorted by highest desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table2_str.to_string()}")
            else:
                logger.info(f"No first words with highest >= {MIN_HIGHEST}% and total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")
                return None

            return table2_num
        except Exception as e:
            logger.exception(f"Failed to build visual relationships: {e}")
            return None

    def build_visual_relationships_2(self, df_group, authors):
        """
        Build tables showing relationships between emoji sequences and authors in a WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for a specific group with 'message_cleaned' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            tuple: (pandas.DataFrame or None, pandas.DataFrame or None) - Numerical DataFrames for table1 and table2.
        """
        MIN_TOTAL = 10
        MIN_HIGHEST = 60  # in percent

        if df_group.empty:
            logger.error("Empty DataFrame provided for building visual relationships_2.")
            return None, None

        try:
            # Extract emoji sequences from message_cleaned
            sequences = []
            for _, row in df_group.iterrows():
                message = row['message_cleaned']
                author = row['author']
                if isinstance(message, str):
                    emoji_sequences = self.emoji_pattern.findall(message)
                    for seq in emoji_sequences:
                        sequences.append({'sequence': seq, 'author': author})

            if not sequences:
                logger.info("No emoji sequences found in the group.")
                return None, None

            seq_df = pd.DataFrame(sequences)
            counts = seq_df.groupby(['sequence', 'author']).size().reset_index(name='count')
            pivot = counts.pivot(index='sequence', columns='author', values='count').fillna(0)

            # Calculate total
            pivot['total'] = pivot.sum(axis=1)

            # Filter out rows where total == 0 (though unlikely)
            pivot = pivot[pivot['total'] > 0]

            # Authors columns
            authors = sorted([a for a in authors if a in pivot.columns])
            if not authors:
                logger.error("No matching authors found in pivot table.")
                return None, None

            # Convert counts to int
            pivot[authors] = pivot[authors].astype(int)
            pivot['total'] = pivot['total'].astype(int)

            # Calculate percentages (numerical)
            percentages = pivot[authors].div(pivot['total'], axis=0) * 100
            percentages['highest'] = percentages.max(axis=1)

            # Count number of authors with non-zero percentages
            non_zero_authors = (percentages[authors] > 0).sum(axis=1)

            # Combine into full numerical table: total + percentages (authors + highest)
            full_table_num = pd.concat([pivot[['total']], percentages], axis=1)

            # Create string version for logging
            full_table_str = full_table_num.copy()
            for col in authors + ['highest']:
                full_table_str[col] = full_table_str[col].apply(lambda x: f"{int(x)}%")

            # Table 1: total >= MIN_TOTAL, sorted by total descending
            table1_num = full_table_num[full_table_num['total'] >= MIN_TOTAL].sort_values('total', ascending=False)
            if not table1_num.empty:
                table1_str = full_table_str.loc[table1_num.index]
                logger.info(f"Table 1 (total >= {MIN_TOTAL}, sorted by total desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table1_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")

            # Table 2: highest >= MIN_HIGHEST, total >= MIN_TOTAL, and max 2 authors with >0%, sorted by highest descending
            table2_num = full_table_num[
                (full_table_num['highest'] >= MIN_HIGHEST) & 
                (full_table_num['total'] >= MIN_TOTAL) & 
                (non_zero_authors <= 2)
            ].sort_values('highest', ascending=False)
            if not table2_num.empty:
                table2_str = full_table_str.loc[table2_num.index]
                logger.info(f"Table 2 (highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, max 2 authors with >0%, sorted by highest desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table2_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, and max 2 authors with >0% for group {df_group['whatsapp_group'].iloc[0]}.")
                return table1_num, None

            return table1_num, table2_num
        except Exception as e:
            logger.exception(f"Failed to build visual relationships_2: {e}")
            return None, None
